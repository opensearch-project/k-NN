/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

import org.apache.lucene.util.ArrayUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT;

/**
 * Writer for half-precision (FP16) flat vector fields. Encodes incoming FP32 vectors to FP16
 * (2 bytes per dimension) and writes them sequentially to a {@code .vec} file, with per-field
 * metadata stored in a {@code .vemf} file.
 *
 * The on-disk layout follows the same structural pattern as Lucene's {@code Lucene99FlatVectorsWriter}:
 * Each float dimension is converted to IEEE 754 half-float and stored as 2 bytes in little-endian order.
 */
public class KNN1040HalfFloatFlatVectorsWriter extends FlatVectorsWriter {

    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(KNN1040HalfFloatFlatVectorsWriter.class);

    private static final int VECTOR_DATA_ALIGNMENT = 64;

    private final SegmentWriteState segmentWriteState;
    private final IndexOutput meta;
    private final IndexOutput vectorData;
    private final List<FieldData> fields = new ArrayList<>();
    private boolean finished;

    private record FieldData(FlatFieldVectorsWriter<?> fieldWriter, FieldInfo fieldInfo) {
    }

    /**
     * Creates a new writer for FP16 flat vectors.
     *
     * @param state  the segment write state
     * @param scorer the flat vectors scorer used for scoring during indexing
     * @throws IOException if an I/O error occurs while creating output files
     */
    public KNN1040HalfFloatFlatVectorsWriter(SegmentWriteState state, FlatVectorsScorer scorer) throws IOException {
        super(scorer);
        this.segmentWriteState = state;

        boolean success = false;
        try {
            String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
            String vectorDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);

            meta = state.directory.createOutput(metaFileName, state.context);
            vectorData = state.directory.createOutput(vectorDataFileName, state.context);

            CodecUtil.writeIndexHeader(meta, META_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            CodecUtil.writeIndexHeader(vectorData, VECTOR_DATA_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        checkFloat32Encoding(fieldInfo);
        // NOTE: FlatFieldVectorsWriter has no public static create() method.
        FlatFieldVectorsWriter<?> fieldWriter = new FloatFieldWriter(fieldInfo);
        fields.add(new FieldData(fieldWriter, fieldInfo));
        return fieldWriter;
    }

    private static void checkFloat32Encoding(FieldInfo fieldInfo) {
        if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new IllegalArgumentException(
                "FP16 flat format only supports FLOAT32 encoding, got ["
                    + fieldInfo.getVectorEncoding()
                    + "] for field ["
                    + fieldInfo.name
                    + "]"
            );
        }
    }

    /**
     * Returns the aligned offset for a field's vector data, mirroring {@code
     * Lucene99FlatVectorsWriter.alignOutput}. Needed because {@link #writeMeta} also writes
     * ord-to-doc data into {@code .vec}, so a later field would otherwise start unaligned.
     */
    private static long alignOutput(IndexOutput output) throws IOException {
        return output.alignFilePointer(VECTOR_DATA_ALIGNMENT);
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        for (FieldData field : fields) {
            if (sortMap == null) {
                writeField(field.fieldWriter(), field.fieldInfo(), maxDoc);
            } else {
                writeSortingField(field.fieldWriter(), field.fieldInfo(), maxDoc, sortMap);
            }
            field.fieldWriter().finish();
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        if (meta != null) {
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (vectorData != null) {
            CodecUtil.writeFooter(vectorData);
        }
    }

    @Override
    public void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        checkFloat32Encoding(fieldInfo);

        // Delegates to MergedVectorValues so vectors come out in final merged-segment doc order,
        // with deletions filtered and index sorting applied. A naive per-reader loop can't
        // reproduce this order: under an index sort, each reader's docs are remapped to
        // non-contiguous ids interleaved with other readers' docs in the merged segment, not laid
        // out reader-by-reader.
        final FloatVectorValues mergedValues = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);

        final long vectorDataOffset = alignOutput(vectorData);
        final DocsWithFieldSet docsWithField = writeVectorData(vectorData, mergedValues, fieldInfo.getVectorDimension());
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

        writeMeta(fieldInfo, segmentWriteState.segmentInfo.maxDoc(), vectorDataOffset, vectorDataLength, docsWithField);
    }

    // Encodes vectors to FP16 and writes them, returning the documents that have a vector.
    private static DocsWithFieldSet writeVectorData(IndexOutput output, FloatVectorValues values, int dimension) throws IOException {
        final byte[] outputBuffer = new byte[dimension * Short.BYTES];
        final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        final KnnVectorValues.DocIndexIterator iterator = values.iterator();
        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            final float[] vector = values.vectorValue(iterator.index());
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, outputBuffer, dimension);
            output.writeBytes(outputBuffer, 0, outputBuffer.length);
            docsWithField.add(doc);
        }
        return docsWithField;
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldData field : fields) {
            total += field.fieldWriter().ramBytesUsed();
        }
        return total;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorData);
    }

    private void writeField(FlatFieldVectorsWriter<?> fieldWriter, FieldInfo fieldInfo, int maxDoc) throws IOException {
        final int dimension = fieldInfo.getVectorDimension();
        final byte[] outputBuffer = new byte[dimension * Short.BYTES];
        @SuppressWarnings("unchecked")
        final List<float[]> vectors = (List<float[]>) fieldWriter.getVectors();

        final long vectorDataOffset = alignOutput(vectorData);
        for (float[] vector : vectors) {
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, outputBuffer, dimension);
            vectorData.writeBytes(outputBuffer, 0, outputBuffer.length);
        }
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

        writeMeta(fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, fieldWriter.getDocsWithFieldSet());
    }

    private void writeSortingField(FlatFieldVectorsWriter<?> fieldWriter, FieldInfo fieldInfo, int maxDoc, Sorter.DocMap sortMap)
        throws IOException {
        final int dimension = fieldInfo.getVectorDimension();
        final byte[] outputBuffer = new byte[dimension * Short.BYTES];
        @SuppressWarnings("unchecked")
        final List<float[]> vectors = (List<float[]>) fieldWriter.getVectors();

        final DocsWithFieldSet docsWithFieldSet = fieldWriter.getDocsWithFieldSet();
        final int[] ordMap = new int[docsWithFieldSet.cardinality()]; // new ord to old ord
        final DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
        mapOldOrdToNewOrd(docsWithFieldSet, sortMap, null, ordMap, newDocsWithField);

        final long vectorDataOffset = alignOutput(vectorData);
        for (int ordinal : ordMap) {
            final float[] vector = vectors.get(ordinal);
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, outputBuffer, dimension);
            vectorData.writeBytes(outputBuffer, 0, outputBuffer.length);
        }
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

        writeMeta(fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, newDocsWithField);
    }

    private void writeMeta(FieldInfo fieldInfo, int maxDoc, long vectorDataOffset, long vectorDataLength, DocsWithFieldSet docsWithField)
        throws IOException {
        meta.writeInt(fieldInfo.number);
        meta.writeInt(fieldInfo.getVectorEncoding().ordinal());
        meta.writeInt(fieldInfo.getVectorSimilarityFunction().ordinal());
        meta.writeVLong(vectorDataOffset);
        meta.writeVLong(vectorDataLength);
        meta.writeVInt(fieldInfo.getVectorDimension());

        // write docIDs
        final int count = docsWithField.cardinality();
        meta.writeInt(count);
        OrdToDocDISIReaderConfiguration.writeStoredMeta(DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);
    }

    /**
     * Per-field writer that stores {@code float[]} on heap during indexing.
     */
    private static class FloatFieldWriter extends FlatFieldVectorsWriter<float[]> {
        private static final long FIELD_WRITER_SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(FloatFieldWriter.class);

        private final FieldInfo fieldInfo;
        private final List<float[]> vectors = new ArrayList<>();
        private final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        private boolean finished;
        private int lastDocID = -1;

        FloatFieldWriter(FieldInfo fieldInfo) {
            this.fieldInfo = fieldInfo;
        }

        @Override
        public void addValue(int docID, float[] vectorValue) throws IOException {
            if (finished) {
                throw new IllegalStateException("already finished");
            }
            if (docID == lastDocID) {
                throw new IllegalArgumentException("VectorValuesField \"" + fieldInfo.name + "\" appears more than once in this document");
            }
            docsWithField.add(docID);
            vectors.add(copyValue(vectorValue));
            lastDocID = docID;
        }

        @Override
        public float[] copyValue(float[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, fieldInfo.getVectorDimension());
        }

        @Override
        public List<float[]> getVectors() {
            return vectors;
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return docsWithField;
        }

        @Override
        public long ramBytesUsed() {
            long size = FIELD_WRITER_SHALLOW_RAM_BYTES_USED;
            if (vectors.isEmpty()) return size;
            return size + docsWithField.ramBytesUsed() + (long) vectors.size() * (RamUsageEstimator.NUM_BYTES_OBJECT_REF
                + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER) + (long) vectors.size() * fieldInfo.getVectorDimension() * fieldInfo
                    .getVectorEncoding().byteSize;
        }

        @Override
        public void finish() throws IOException {
            finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished;
        }
    }
}
