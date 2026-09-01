/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
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
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext.FileOpenHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import java.io.IOException;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.BULK_SCORE_BATCH_SIZE;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VERSION_START;

/**
 * Reader for half-precision (FP16) flat vector fields, reading vectors from {@code .vec} and
 * per-field metadata from {@code .vemf}.
 *
 * Scoring is selected by {@link KNN1040HalfFloatFlatVectorsValues#selectScorer}: native SIMD
 * (mmap zero-copy when available, else a heap-buffer copy) when the similarity has a native type,
 * otherwise a plain Java fallback.
 */
@Log4j2
public class KNN1040HalfFloatFlatVectorsReader extends FlatVectorsReader {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(KNN1040HalfFloatFlatVectorsReader.class);

    private static final String VECTOR_VALUES_SLICE = "KNN1040HalfFloatFlatVectorsValuesSlice";

    private final IntObjectHashMap<FieldEntry> fields = new IntObjectHashMap<>();
    private final IndexInput vectorData;
    private final FieldInfos fieldInfos;
    private final FlatVectorsScorer scorer;
    private final IOContext dataContext;

    public KNN1040HalfFloatFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
        this(state, scorer, DataAccessHint.RANDOM);
    }

    private static FileOpenHint[] buildFileOpenHints(DataAccessHint accessHint) {
        return accessHint == null
            ? new FileOpenHint[] { FileTypeHint.DATA, FileDataHint.KNN_VECTORS }
            : new FileOpenHint[] { FileTypeHint.DATA, FileDataHint.KNN_VECTORS, accessHint };
    }

    public KNN1040HalfFloatFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer, DataAccessHint accessHint)
        throws IOException {
        super();
        this.scorer = scorer;
        this.fieldInfos = state.fieldInfos;
        this.dataContext = state.context.withHints(buildFileOpenHints(accessHint));

        boolean success = false;
        try {
            int versionMeta = readMetadata(state);
            vectorData = openDataInput(state, versionMeta, VECTOR_DATA_EXTENSION, VECTOR_DATA_CODEC_NAME, dataContext);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
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
        boolean success = false;
        try {
            int versionVectorData = CodecUtil.checkIndexHeader(
                in,
                codecName,
                VERSION_START,
                VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionVectorData) {
                throw new CorruptIndexException(
                    String.format(Locale.ROOT, "Format versions mismatch: meta=%d, %s=%d", versionMeta, codecName, versionVectorData),
                    in
                );
            }
            CodecUtil.retrieveChecksum(in);
            success = true;
            return in;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(in);
            }
        }
    }

    private int readMetadata(SegmentReadState state) throws IOException {
        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
        int versionMeta;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    meta,
                    META_CODEC_NAME,
                    VERSION_START,
                    VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(meta);
            } catch (Throwable t) {
                priorE = t;
                throw t;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
        }
        return versionMeta;
    }

    private void readFields(ChecksumIndexInput meta) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }
            fields.put(info.number, FieldEntry.create(meta, info));
        }
    }

    // A whitelist, not a decode table: readSimilarityFunction decodes via VectorSimilarityFunction
    // .values()[i] directly (the same ordinal the writer persists), then checks membership here so an
    // ordinal for a similarity function this codec hasn't validated FP16 scoring for is rejected
    // instead of silently accepted.
    private static final Set<VectorSimilarityFunction> SUPPORTED_SIMILARITY_FUNCTIONS = EnumSet.of(
        VectorSimilarityFunction.EUCLIDEAN,
        VectorSimilarityFunction.DOT_PRODUCT,
        VectorSimilarityFunction.COSINE,
        VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
    );

    private static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
        int i = input.readInt();
        VectorSimilarityFunction[] values = VectorSimilarityFunction.values();
        if (i < 0 || i >= values.length || !SUPPORTED_SIMILARITY_FUNCTIONS.contains(values[i])) {
            throw new CorruptIndexException("invalid distance function: " + i, input);
        }
        return values[i];
    }

    private static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
        int encodingId = input.readInt();
        if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
            throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
        }
        return VectorEncoding.values()[encodingId];
    }

    private FieldEntry getFieldEntryOrThrow(String field) {
        final FieldInfo info = fieldInfos.fieldInfo(field);
        final FieldEntry entry;
        if (info == null || (entry = fields.get(info.number)) == null) {
            throw new IllegalArgumentException("field=\"" + field + "\" not found");
        }
        return entry;
    }

    private FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
        final FieldEntry fieldEntry = getFieldEntryOrThrow(field);
        if (fieldEntry.vectorEncoding != expectedEncoding) {
            throw new IllegalArgumentException(
                "field=\"" + field + "\" is encoded as: " + fieldEntry.vectorEncoding + " expected: " + expectedEncoding
            );
        }
        return fieldEntry;
    }

    /**
     * Builds a fresh {@link KNN1040HalfFloatFlatVectorsValues} over {@code entry}'s slice of the
     * {@code .vec} file.
     */
    private KNN1040HalfFloatFlatVectorsValues newVectorValues(FieldEntry entry) throws IOException {
        IndexInput slice = vectorData.slice(VECTOR_VALUES_SLICE, entry.vectorDataOffset, entry.vectorDataLength);
        boolean needsOrdToDocReader = entry.ordToDoc.isDense() == false && entry.ordToDoc.isEmpty() == false;
        DirectMonotonicReader ordToDocReader = needsOrdToDocReader ? entry.ordToDoc.getDirectMonotonicReader(vectorData) : null;
        return new KNN1040HalfFloatFlatVectorsValues(entry.dimension, entry.size, slice, ordToDocReader, entry.similarity);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FieldEntry entry = getFieldEntry(field, VectorEncoding.FLOAT32);
        KNN1040HalfFloatFlatVectorsValues base = newVectorValues(entry);
        long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(base.getSlice(), 0, base.getSlice().length());
        return addressAndSize != null ? new MMapFloatVectorValues(base, addressAndSize) : base;
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        final FieldEntry entry = getFieldEntry(field, VectorEncoding.FLOAT32);
        KNN1040HalfFloatFlatVectorsValues base = newVectorValues(entry);
        if (base.size() == 0) {
            return null;
        }
        return KNN1040HalfFloatFlatVectorsValues.selectScorer(base, target, entry.similarity);
    }

    // Exhaustive brute-force search over all FP16 vectors, scoring ords in batches.
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer randomScorer = getRandomVectorScorer(field, target);
        if (randomScorer == null) {
            return;
        }

        int numVectors = randomScorer.maxOrd();
        if (numVectors == 0 || knnCollector.k() == 0) {
            return;
        }

        final Bits acceptedOrds = randomScorer.getAcceptOrds(acceptDocs.bits());
        int[] ords = new int[BULK_SCORE_BATCH_SIZE];
        float[] scores = new float[BULK_SCORE_BATCH_SIZE];
        int numOrds = 0;

        for (int i = 0; i < numVectors; i++) {
            if (knnCollector.earlyTerminated()) {
                break;
            }
            if (acceptedOrds == null || acceptedOrds.get(i)) {
                ords[numOrds++] = i;
                if (numOrds == BULK_SCORE_BATCH_SIZE) {
                    collectBatch(randomScorer, knnCollector, ords, scores, numOrds);
                    numOrds = 0;
                }
            }
        }

        if (numOrds > 0) {
            collectBatch(randomScorer, knnCollector, ords, scores, numOrds);
        }
    }

    private void collectBatch(RandomVectorScorer scorer, KnnCollector knnCollector, int[] ords, float[] scores, int numOrds)
        throws IOException {
        knnCollector.incVisitedCount(numOrds);
        if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
            for (int j = 0; j < numOrds; j++) {
                knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
            }
        }
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("FP16 format does not support byte vectors");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        throw new UnsupportedOperationException("FP16 format does not support byte vector scoring");
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        getFieldEntryOrThrow(field);
        return scorer;
    }

    @Override
    public FlatVectorsReader getMergeInstance() throws IOException {
        vectorData.updateIOContext(dataContext.withHints(DataAccessHint.SEQUENTIAL));
        return this;
    }

    @Override
    public void checkIntegrity() throws IOException {
        CodecUtil.checksumEntireFile(vectorData);
    }

    @Override
    public long ramBytesUsed() {
        return KNN1040HalfFloatFlatVectorsReader.SHALLOW_SIZE + fields.ramBytesUsed();
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        final FieldEntry entry = getFieldEntryOrThrow(fieldInfo.name);
        return Map.of(VECTOR_DATA_EXTENSION, entry.vectorDataLength());
    }

    @Override
    public void finishMerge() throws IOException {
        vectorData.updateIOContext(dataContext);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(vectorData);
    }

    private record FieldEntry(VectorSimilarityFunction similarity, VectorEncoding vectorEncoding, long vectorDataOffset,
        long vectorDataLength, int dimension, int size, OrdToDocDISIReaderConfiguration ordToDoc, FieldInfo info) {

        FieldEntry {
            if (vectorEncoding != VectorEncoding.FLOAT32) {
                throw new IllegalStateException(
                    "Unexpected vector encoding for field=\"" + info.name + "\"; expected FLOAT32, got " + vectorEncoding
                );
            }
            if (similarity != info.getVectorSimilarityFunction()) {
                throw new IllegalStateException(
                    "Inconsistent vector similarity function for field=\""
                        + info.name
                        + "\"; "
                        + similarity
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

            final int byteSize = Short.BYTES;
            long vectorBytes = Math.multiplyExact((long) infoVectorDimension, byteSize);
            long numBytes = Math.multiplyExact(vectorBytes, size);
            if (numBytes != vectorDataLength) {
                throw new IllegalStateException(
                    "Vector data length "
                        + vectorDataLength
                        + " not matching size="
                        + size
                        + " * dim="
                        + dimension
                        + " * byteSize="
                        + byteSize
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
