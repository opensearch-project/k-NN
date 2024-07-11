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

import lombok.Getter;
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
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.opensearch.knn.index.NativeIndexCreationManager;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
    @SuppressWarnings("unchecked")
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
            NativeIndexCreationManager.startIndexCreation(
                segmentWriteState,
                KNNVectorValuesFactory.getFloatVectorValues(fieldWriter),
                fieldWriter.fieldInfo
            );
        }
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // This will ensure that we are merging the FlatIndex during force merge.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
        final FloatVectorValues floatVectorValues = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        // merging the graphs here
        NativeIndexCreationManager.startIndexCreation(
            segmentWriteState,
            KNNVectorValuesFactory.getFloatVectorValues(floatVectorValues),
            fieldInfo
        );
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

    public static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private final FieldInfo fieldInfo;
        @Getter
        private final Map<Integer, T> vectorsMap;
        private int lastDocID = -1;
        @Getter
        private final DocsWithFieldSet docsWithField;

        private final InfoStream infoStream;

        static NativeEnginesKNNVectorsWriter.FieldWriter<?> create(FieldInfo fieldInfo, InfoStream infoStream) {
            return new NativeEnginesKNNVectorsWriter.FieldWriter<float[]>(fieldInfo, infoStream);
        }

        FieldWriter(final FieldInfo fieldInfo, final InfoStream infoStream) {
            this.fieldInfo = fieldInfo;
            this.infoStream = infoStream;
            vectorsMap = new HashMap<>();
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
            vectorsMap.put(docID, vectorValue);
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
