/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

import java.io.IOException;
import java.io.UncheckedIOException;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNIOUtils;

/**
 * A FlatVectorsReader that reads half-precision (2-byte) FP16 data from .vec files,
 * decodes to float32 via the KNN serializer, and otherwise follows Lucene99 logic.
 */
public final class KNN990HalfFloatFlatVectorsReader extends FlatVectorsReader {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(KNN990HalfFloatFlatVectorsReader.class);
    private static final KNNVectorAsCollectionOfHalfFloatsSerializer SERIALIZER = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;

    private final IntObjectHashMap<FieldEntry> fields = new IntObjectHashMap<>();
    private final IndexInput vectorData;
    private final FieldInfos fieldInfos;

    public KNN990HalfFloatFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
        super(scorer);
        int versionMeta = readMetadata(state);
        this.fieldInfos = state.fieldInfos;
        try {
            vectorData = openDataInput(
                state,
                versionMeta,
                KNN990HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION,
                KNN990HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                // Flat formats are used to randomly access vectors from their node ID that is stored
                // in the HNSW graph.
                IOContext.DEFAULT
            );
        } catch (Throwable t) {
            KNNIOUtils.closeWhileSuppressingExceptions(t, this);
            throw t;
        }
    }

    private int readMetadata(SegmentReadState state) throws IOException {
        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN990HalfFloatFlatVectorsFormat.META_EXTENSION
        );
        int versionMeta = -1;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    meta,
                    KNN990HalfFloatFlatVectorsFormat.META_CODEC_NAME,
                    KNN990HalfFloatFlatVectorsFormat.VERSION_START,
                    KNN990HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(meta, state.fieldInfos);
            } catch (Throwable exception) {
                priorE = exception;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
        }
        return versionMeta;
    }

    private static IndexInput openDataInput(
        SegmentReadState state,
        int versionMeta,
        String fileExtension,
        String codecName,
        IOContext context
    ) throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
        IndexInput in = state.directory.openInput(fileName, context);
        try {
            int versionVectorData = CodecUtil.checkIndexHeader(
                in,
                codecName,
                KNN990HalfFloatFlatVectorsFormat.VERSION_START,
                KNN990HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionVectorData) {
                throw new CorruptIndexException(
                    "Format versions mismatch: meta=" + versionMeta + ", " + codecName + "=" + versionVectorData,
                    in
                );
            }
            CodecUtil.retrieveChecksum(in);
            return in;
        } catch (Throwable t) {
            KNNIOUtils.closeWhileSuppressingExceptions(t, in);
            throw t;
        }
    }

    private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            FieldInfo info = infos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }
            KNN990HalfFloatFlatVectorsReader.FieldEntry fieldEntry = KNN990HalfFloatFlatVectorsReader.FieldEntry.create(meta, info);
            fields.put(info.number, fieldEntry);
        }
    }

    @Override
    public long ramBytesUsed() {
        return KNN990HalfFloatFlatVectorsReader.SHALLOW_SIZE + fields.ramBytesUsed();
    }

    @Override
    public void checkIntegrity() throws IOException {
        CodecUtil.checksumEntireFile(vectorData);
    }

    @Override
    public FlatVectorsReader getMergeInstance() {
        try {
            // Update the read advice since vectors are guaranteed to be accessed sequentially for merge
            this.vectorData.updateReadAdvice(ReadAdvice.SEQUENTIAL);
            return this;
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    private FieldEntry getFieldEntryOrThrow(String field) {
        final FieldInfo info = fieldInfos.fieldInfo(field);
        final FieldEntry entry;
        if (info == null || (entry = fields.get(info.number)) == null) {
            throw new IllegalArgumentException("field=\"" + field + "\" not found");
        }
        return entry;
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        final FieldEntry fe = getFieldEntryOrThrow(field);

        OffHeapByteVectorValues base = OffHeapByteVectorValues.load(
            fe.similarityFunction,
            vectorScorer,
            fe.ordToDoc,
            VectorEncoding.BYTE,
            fe.dimension,
            fe.vectorDataOffset,
            fe.vectorDataLength,
            vectorData
        );

        final int dim = fe.dimension;
        final int byteSize = dim * Short.BYTES;

        return new ByteVectorValues() {
            private final byte[] bytesBuffer = new byte[byteSize];
            private final IndexInput slice = base.getSlice();

            @Override
            public int dimension() {
                return dim;
            }

            @Override
            public int size() {
                return base.size();
            }

            @Override
            public int ordToDoc(int ord) {
                return base.ordToDoc(ord);
            }

            @Override
            public Bits getAcceptOrds(Bits bits) {
                return base.getAcceptOrds(bits);
            }

            @Override
            public DocIndexIterator iterator() {
                return base.iterator();
            }

            @Override
            public byte[] vectorValue(int ord) throws IOException {
                slice.seek((long) ord * byteSize);
                slice.readBytes(bytesBuffer, 0, bytesBuffer.length);
                return bytesBuffer;
            }

            @Override
            public ByteVectorValues copy() {
                return this;
            }

            @Override
            public VectorScorer scorer(byte[] query) throws IOException {
                return base.scorer(query);
            }
        };
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FieldEntry fe = getFieldEntryOrThrow(field);

        OffHeapFloatVectorValues base = OffHeapFloatVectorValues.load(
            fe.similarityFunction,
            vectorScorer,
            fe.ordToDoc,
            fe.vectorEncoding,
            fe.dimension,
            fe.vectorDataOffset,
            fe.vectorDataLength,
            vectorData
        );

        final int dim = fe.dimension;
        final int byteSize = dim * Short.BYTES;

        return new FloatVectorValues() {
            private final byte[] bytesBuffer = new byte[byteSize];
            private final float[] floatBuffer = new float[dim];
            private final IndexInput slice = base.getSlice();

            @Override
            public int dimension() {
                return dim;
            }

            @Override
            public int size() {
                return base.size();
            }

            @Override
            public int ordToDoc(int ord) {
                return base.ordToDoc(ord);
            }

            @Override
            public Bits getAcceptOrds(Bits bits) {
                return base.getAcceptOrds(bits);
            }

            @Override
            public DocIndexIterator iterator() {
                return base.iterator();
            }

            @Override
            public float[] vectorValue(int ord) throws IOException {
                slice.seek((long) ord * byteSize);
                slice.readBytes(bytesBuffer, 0, bytesBuffer.length);
                SERIALIZER.byteToFloatArray(bytesBuffer, floatBuffer, dim, 0);
                return floatBuffer;
            }

            @Override
            public FloatVectorValues copy() {
                return this;
            }

            @Override
            public VectorScorer scorer(float[] query) throws IOException {
                return base.scorer(query);
            }
        };
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        FloatVectorValues vals = getFloatVectorValues(field);
        FieldEntry fe = getFieldEntryOrThrow(field);
        return vectorScorer.getRandomVectorScorer(fe.similarityFunction, vals, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        throw new UnsupportedOperationException("HalfFloatFlatVectorsReader does not support byte[] targets");
    }

    @Override
    public void finishMerge() throws IOException {
        // This makes sure that the access pattern hint is reverted back since HNSW implementation
        // needs it
        this.vectorData.updateReadAdvice(ReadAdvice.RANDOM);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(vectorData);
    }

    private record FieldEntry(VectorSimilarityFunction similarityFunction, VectorEncoding vectorEncoding, long vectorDataOffset,
        long vectorDataLength, int dimension, int size, OrdToDocDISIReaderConfiguration ordToDoc, FieldInfo info) {

        FieldEntry {
            if (similarityFunction != info.getVectorSimilarityFunction()) {
                throw new IllegalStateException(
                    "Inconsistent vector similarity function for field=\""
                        + info.name
                        + "\"; "
                        + similarityFunction
                        + " != "
                        + info.getVectorSimilarityFunction()
                );
            }
            int infoVectorDimension = info.getVectorDimension();
            if (infoVectorDimension != dimension) {
                throw new IllegalStateException(
                    "Inconsistent vector dimension for field=\"" + info.name + "\"; " + infoVectorDimension + " != " + dimension
                );
            }

            int byteSizeize = Short.BYTES;
            long vectorBytes = Math.multiplyExact((long) infoVectorDimension, byteSizeize);
            long numBytes = Math.multiplyExact(vectorBytes, size);
            if (numBytes != vectorDataLength) {
                throw new IllegalStateException(
                    "Vector data length "
                        + vectorDataLength
                        + " not matching size="
                        + size
                        + " * dim="
                        + dimension
                        + " * byteSizeize="
                        + byteSizeize
                        + " = "
                        + numBytes
                );
            }
        }

        static FieldEntry create(IndexInput input, FieldInfo info) throws IOException {
            final VectorEncoding vectorEncoding = readVectorEncoding(input);
            final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
            final var vectorDataOffset = input.readVLong();
            final var vectorDataLength = input.readVLong();
            final var dimension = input.readVInt();
            final var size = input.readInt();
            final var ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(input, size);
            return new FieldEntry(similarityFunction, vectorEncoding, vectorDataOffset, vectorDataLength, dimension, size, ordToDoc, info);
        }
    }
}
