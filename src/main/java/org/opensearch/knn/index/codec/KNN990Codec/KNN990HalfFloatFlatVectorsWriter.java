/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import static org.opensearch.knn.index.codec.KNN990Codec.KNN990HalfFloatFlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNIOUtils;

/**
 * Writes half float vector values to index segments.
 */
public final class KNN990HalfFloatFlatVectorsWriter extends FlatVectorsWriter {

    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(KNN990HalfFloatFlatVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final IndexOutput meta, vectorData;

    private final List<FieldWriter<?>> fields = new ArrayList<>();
    private boolean finished;

    public KNN990HalfFloatFlatVectorsWriter(SegmentWriteState state, FlatVectorsScorer scorer) throws IOException {
        super(scorer);
        this.segmentWriteState = state;
        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN990HalfFloatFlatVectorsFormat.META_EXTENSION
        );

        String vectorDataFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN990HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION
        );

        try {
            meta = state.directory.createOutput(metaFileName, state.context);
            vectorData = state.directory.createOutput(vectorDataFileName, state.context);

            CodecUtil.writeIndexHeader(
                meta,
                KNN990HalfFloatFlatVectorsFormat.META_CODEC_NAME,
                KNN990HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            CodecUtil.writeIndexHeader(
                vectorData,
                KNN990HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                KNN990HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
        } catch (Throwable t) {
            KNNIOUtils.closeWhileSuppressingExceptions(t, this);
            throw t;
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        FieldWriter<?> newField = FieldWriter.create(fieldInfo);
        fields.add(newField);
        return newField;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        for (FieldWriter<?> field : fields) {
            if (sortMap == null) {
                writeField(field, maxDoc);
            } else {
                writeSortingField(field, maxDoc, sortMap);
            }
            field.finish();
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (vectorData != null) {
            CodecUtil.writeFooter(vectorData);
        }
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldWriter<?> field : fields) {
            total += field.ramBytesUsed();
        }
        return total;
    }

    private void writeField(FieldWriter<?> fieldData, int maxDoc) throws IOException {
        long vectorDataOffset = vectorData.alignFilePointer(Short.BYTES);
        int dim = fieldData.fieldInfo.getVectorDimension();
        KNNVectorAsCollectionOfHalfFloatsSerializer vectorSerializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(dim);

        for (Object v : fieldData.vectors) {
            byte[] vector = vectorSerializer.floatToByteArray((float[]) v);
            vectorData.writeBytes(vector, vector.length);
        }
        long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, fieldData.docsWithField);
    }

    private void writeSortingField(FieldWriter<?> fieldData, int maxDoc, Sorter.DocMap sortMap) throws IOException {
        final int[] ordMap = new int[fieldData.docsWithField.cardinality()]; // new ord to old ord

        DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
        mapOldOrdToNewOrd(fieldData.docsWithField, sortMap, null, ordMap, newDocsWithField);

        long vectorDataOffset = vectorData.alignFilePointer(Short.BYTES);
        int dim = fieldData.fieldInfo.getVectorDimension();
        KNNVectorAsCollectionOfHalfFloatsSerializer vectorSerializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(dim);

        for (int ordinal : ordMap) {
            byte[] vector = vectorSerializer.floatToByteArray((float[]) fieldData.vectors.get(ordinal));
            vectorData.writeBytes(vector, vector.length);
        }

        long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

        writeMeta(fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, newDocsWithField);
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Since we know we will not be searching for additional indexing, we can just write
        // the vectors directly to the new segment.
        long vectorDataOffset = vectorData.alignFilePointer(Short.BYTES);
        // No need to use temporary file as we don't have to re-open for reading
        DocsWithFieldSet docsWithField = writeHalfFloatVectorData(
            vectorData,
            KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState)
        );
        long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldInfo, segmentWriteState.segmentInfo.maxDoc(), vectorDataOffset, vectorDataLength, docsWithField);
    }

    /**
     * Writes the half float vector values to the output and returns a set of documents that contains vectors.
     */
    private static DocsWithFieldSet writeHalfFloatVectorData(IndexOutput output, FloatVectorValues floatVectorValues) throws IOException {
        DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        KnnVectorValues.DocIndexIterator iter = floatVectorValues.iterator();

        int dim = floatVectorValues.dimension();
        KNNVectorAsCollectionOfHalfFloatsSerializer vectorSerializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(dim);

        for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
            float[] value = floatVectorValues.vectorValue(iter.index());
            byte[] half = vectorSerializer.floatToByteArray(value);
            output.writeBytes(half, half.length);
            docsWithField.add(docV);
        }
        return docsWithField;
    }

    @Override
    public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        throw new UnsupportedOperationException("Lucene ANN not implemented yet");
    }

    private void writeMeta(FieldInfo field, int maxDoc, long vectorDataOffset, long vectorDataLength, DocsWithFieldSet docsWithField)
        throws IOException {
        meta.writeInt(field.number);
        meta.writeInt(VectorEncoding.FLOAT32.ordinal());
        meta.writeInt(field.getVectorSimilarityFunction().ordinal());
        meta.writeVLong(vectorDataOffset);
        meta.writeVLong(vectorDataLength);
        meta.writeVInt(field.getVectorDimension());
        int count = docsWithField.cardinality();
        meta.writeInt(count);
        OrdToDocDISIReaderConfiguration.writeStoredMeta(DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorData);
    }

    private abstract static class FieldWriter<T> extends FlatFieldVectorsWriter<T> {
        private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
        private final FieldInfo fieldInfo;
        private final int dim;
        private final DocsWithFieldSet docsWithField;
        private final List<T> vectors;
        private boolean finished;

        private int lastDocID = -1;

        static FieldWriter<?> create(FieldInfo fieldInfo) {
            int dim = fieldInfo.getVectorDimension();
            return new KNN990HalfFloatFlatVectorsWriter.FieldWriter<float[]>(fieldInfo) {
                @Override
                public float[] copyValue(float[] value) {
                    return ArrayUtil.copyOfSubArray(value, 0, dim);
                }
            };
        }

        FieldWriter(FieldInfo fieldInfo) {
            super();
            this.fieldInfo = fieldInfo;
            this.dim = fieldInfo.getVectorDimension();
            this.docsWithField = new DocsWithFieldSet();
            vectors = new ArrayList<>();
        }

        @Override
        public void addValue(int docID, T vectorValue) throws IOException {
            if (finished) {
                throw new IllegalStateException("already finished, cannot add more values");
            }
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                    "VectorValuesField \""
                        + fieldInfo.name
                        + "\" appears more than once in this document (only one value is allowed per field)"
                );
            }
            assert docID > lastDocID;
            T copy = copyValue(vectorValue);
            docsWithField.add(docID);
            vectors.add(copy);
            lastDocID = docID;
        }

        @Override
        public long ramBytesUsed() {
            long size = SHALLOW_RAM_BYTES_USED;
            if (vectors.size() == 0) return size;
            return size + docsWithField.ramBytesUsed() + (long) vectors.size() * (RamUsageEstimator.NUM_BYTES_OBJECT_REF
                + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER) + (long) vectors.size() * fieldInfo.getVectorDimension() * Short.BYTES;
        }

        @Override
        public List<T> getVectors() {
            return vectors;
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return docsWithField;
        }

        @Override
        public void finish() throws IOException {
            if (finished) {
                return;
            }
            this.finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished;
        }
    }

    static final class FlatCloseableRandomVectorScorerSupplier implements CloseableRandomVectorScorerSupplier {

        private final RandomVectorScorerSupplier supplier;
        private final Closeable onClose;
        private final int numVectors;

        FlatCloseableRandomVectorScorerSupplier(Closeable onClose, int numVectors, RandomVectorScorerSupplier supplier) {
            this.onClose = onClose;
            this.supplier = supplier;
            this.numVectors = numVectors;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() throws IOException {
            return supplier.scorer();
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return supplier.copy();
        }

        @Override
        public void close() throws IOException {
            onClose.close();
        }

        @Override
        public int totalVectorCount() {
            return numVectors;
        }
    }
}
