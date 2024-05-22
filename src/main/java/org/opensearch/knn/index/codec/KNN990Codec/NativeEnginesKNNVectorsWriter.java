/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesConsumer;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@RequiredArgsConstructor
@Log4j2
public class NativeEnginesKNNVectorsWriter extends KnnVectorsWriter {
    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private final List<NativeEnginesKNNVectorsWriter.FieldWriter<?>> fields = new ArrayList<>();
    private boolean finished;

    /**
     * Add new field for indexing.
     * In Lucene, we use single file for all the vector fields so here we need to see how we are going to make things
     * work.
     * @param fieldInfo {@link FieldInfo}
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        final NativeEnginesKNNVectorsWriter.FieldWriter<?> newField = NativeEnginesKNNVectorsWriter.FieldWriter.create(
            fieldInfo,
            segmentWriteState.infoStream
        );
        // TODO: we can build the graph here too iteratively. but right now I am skipping that as we need iterative
        // graph build support on the JNI layer.
        fields.add(newField);
        return flatVectorsWriter.addField(fieldInfo, newField);
    }

    /**
     * Flush all buffered data on disk. This is not fsync. This is lucene flush.
     *
     * @param maxDoc int
     * @param sortMap {@link Sorter.DocMap}
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        // simply write data in the flat file
        flatVectorsWriter.flush(maxDoc, sortMap);
        // Now let's create the graph here.
        // This is more like a refresh. This is not Opensearch flush.
        // we will use the old school way of creating the graphs here, which was there in BinaryDocValues. Once we
        // start building the graphs iteratively, this function will just be writing data on the file. and storing it
        // on the disk.
        // getFloatsFromFloatVectorValues(fields);
        for (NativeEnginesKNNVectorsWriter.FieldWriter<?> fieldWriter : fields) {
            KNNCodecUtil.Pair pair = getFloatsFromFieldWriter(fieldWriter);
            if (pair.getVectorAddress() == 0 || pair.docs.length == 0) {
                log.info("Skipping engine index creation as there are no vectors or docs in the segment");
                continue;
            }
            KNN80DocValuesConsumer.createNativeIndex(segmentWriteState, fieldWriter.fieldInfo, pair);
        }
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // This will ensure that we are merging the FlatIndex during force merge.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
        final FloatVectorValues floatVectorValues = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        // merging the graphs here
        final KNNCodecUtil.Pair pair = getFloatsFromFloatVectorValues(floatVectorValues);
        KNN80DocValuesConsumer.createNativeIndex(segmentWriteState, fieldInfo, pair);
    }

    /**
     * Called once at the end before close
     */
    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        flatVectorsWriter.finish();
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     *
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter);
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
        return 0;
    }

    private KNNCodecUtil.Pair getFloatsFromFloatVectorValues(FloatVectorValues floatVectorValues) throws IOException {
        List<float[]> vectorList = new ArrayList<>();
        List<Integer> docIdList = new ArrayList<>();
        long vectorAddress = 0;
        int dimension = 0;

        long totalLiveDocs = floatVectorValues.size();
        long vectorsStreamingMemoryLimit = KNNSettings.getVectorStreamingMemoryLimit().getBytes();
        long vectorsPerTransfer = Integer.MIN_VALUE;

        for (int doc = floatVectorValues.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = floatVectorValues.nextDoc()) {
            float[] temp = floatVectorValues.vectorValue();
            // This temp object and copy of temp object is required because when we map floats we read to a memory
            // location in heap always for floatVectorValues. Ref: OffHeapFloatVectorValues.vectorValue.
            float[] vector = Arrays.copyOf(floatVectorValues.vectorValue(), temp.length);
            dimension = vector.length;
            if (vectorsPerTransfer == Integer.MIN_VALUE) {
                vectorsPerTransfer = (dimension * Float.BYTES * totalLiveDocs) / vectorsStreamingMemoryLimit;
                // This condition comes if vectorsStreamingMemoryLimit is higher than total number floats to transfer
                // Doing this will reduce 1 extra trip to JNI layer.
                if (vectorsPerTransfer == 0) {
                    vectorsPerTransfer = totalLiveDocs;
                }
            }

            if (vectorList.size() == vectorsPerTransfer) {
                vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
                // We should probably come up with a better way to reuse the vectorList memory which we have
                // created. Problem here is doing like this can lead to a lot of list memory which is of no use and
                // will be garbage collected later on, but it creates pressure on JVM. We should revisit this.
                vectorList = new ArrayList<>();
            }
            vectorList.add(vector);
            docIdList.add(doc);
        }

        if (vectorList.isEmpty() == false) {
            vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
        }
        // SerializationMode.COLLECTION_OF_FLOATS is not getting used. I just added it to ensure code successfully
        // works.
        return new KNNCodecUtil.Pair(
            docIdList.stream().mapToInt(Integer::intValue).toArray(),
            vectorAddress,
            dimension,
            SerializationMode.COLLECTION_OF_FLOATS
        );
    }

    private KNNCodecUtil.Pair getFloatsFromFieldWriter(NativeEnginesKNNVectorsWriter.FieldWriter<?> fieldWriter) throws IOException {
        List<float[]> vectorList = new ArrayList<>();
        List<Integer> docIdList = new ArrayList<>();
        long vectorAddress = 0;
        int dimension = 0;

        long totalLiveDocs = fieldWriter.vectors.size();
        long vectorsStreamingMemoryLimit = KNNSettings.getVectorStreamingMemoryLimit().getBytes();
        long vectorsPerTransfer = Integer.MIN_VALUE;

        DocIdSetIterator disi = fieldWriter.docsWithField.iterator();

        for (int i = 0; i < fieldWriter.vectors.size(); i++) {
            float[] vector = (float[]) fieldWriter.vectors.get(i);
            dimension = vector.length;
            if (vectorsPerTransfer == Integer.MIN_VALUE) {
                vectorsPerTransfer = (dimension * Float.BYTES * totalLiveDocs) / vectorsStreamingMemoryLimit;
                // This condition comes if vectorsStreamingMemoryLimit is higher than total number floats to transfer
                // Doing this will reduce 1 extra trip to JNI layer.
                if (vectorsPerTransfer == 0) {
                    vectorsPerTransfer = totalLiveDocs;
                }
            }

            if (vectorList.size() == vectorsPerTransfer) {
                vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
                // We should probably come up with a better way to reuse the vectorList memory which we have
                // created. Problem here is doing like this can lead to a lot of list memory which is of no use and
                // will be garbage collected later on, but it creates pressure on JVM. We should revisit this.
                vectorList = new ArrayList<>();
            }

            vectorList.add(vector);
            docIdList.add(disi.nextDoc());

        }

        if (vectorList.isEmpty() == false) {
            vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
        }
        // SerializationMode.COLLECTION_OF_FLOATS is not getting used. I just added it to ensure code successfully
        // works.
        return new KNNCodecUtil.Pair(
            docIdList.stream().mapToInt(Integer::intValue).toArray(),
            vectorAddress,
            dimension,
            SerializationMode.COLLECTION_OF_FLOATS
        );
    }

    private static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private final FieldInfo fieldInfo;
        private final List<T> vectors;
        private int lastDocID = -1;
        private final DocsWithFieldSet docsWithField;

        private final InfoStream infoStream;

        static NativeEnginesKNNVectorsWriter.FieldWriter<?> create(FieldInfo fieldInfo, InfoStream infoStream) {
            return new NativeEnginesKNNVectorsWriter.FieldWriter<float[]>(fieldInfo, infoStream);
        }

        FieldWriter(final FieldInfo fieldInfo, final InfoStream infoStream) {
            this.fieldInfo = fieldInfo;
            this.infoStream = infoStream;
            vectors = new ArrayList<>();
            this.docsWithField = new DocsWithFieldSet();
        }

        /**
         * Add new docID with its vector value to the given field for indexing. Doc IDs must be added in
         * increasing order.
         *
         * @param docID
         * @param vectorValue
         */
        @Override
        public void addValue(int docID, T vectorValue) {
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                    "[NativeEngineKNNVectorWriter]VectorValuesField \""
                        + fieldInfo.name
                        + "\" appears more than once in this document (only one value is allowed per field)"
                );
            }
            assert docID > lastDocID;
            vectors.add(vectorValue);
            docsWithField.add(docID);
            lastDocID = docID;

        }

        /**
         * Used to copy values being indexed to internal storage.
         *
         * @param vectorValue an array containing the vector value to add
         * @return a copy of the value; a new array
         */
        @Override
        public T copyValue(T vectorValue) {
            throw new UnsupportedOperationException();
        }

        /**
         * Return the memory usage of this object in bytes. Negative values are illegal.
         */
        @Override
        public long ramBytesUsed() {
            return 0;
        }

    }

}
