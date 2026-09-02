/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.jni.SimdFp16;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;

/**
 * Decodes FP16 (2 bytes/dimension) vector bytes to {@code float[]} on access.
 *
 * Implements {@link HasIndexSlice} so that {@link org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues}
 * can expose the underlying slice for I/O prefetching. Callers must not hand an instance of this
 * class directly to Lucene's own flat vector scorer factory ({@code Lucene99FlatVectorsScorer}):
 * that factory independently detects {@code HasIndexSlice} and reads the raw slice assuming
 * 4 bytes/dimension (float32), which silently overreads past the buffer on this FP16 (2 bytes/dimension)
 * data. Use {@link #selectNativeOrJavaScorer} instead, which never hands {@code values} to a
 * {@code FlatVectorsScorer} delegate chain.
 */
@Log4j2
class KNN1040HalfFloatFlatVectorsValues extends FloatVectorValues implements HasIndexSlice {
    private final int dimension;
    private final int size;
    private final IndexInput slice;
    private final int byteSize;
    private final float[] value;
    private final byte[] rawBytes;
    private final DirectMonotonicReader ordToDocReader;
    private final VectorSimilarityFunction similarity;
    private int lastOrd = -1;

    KNN1040HalfFloatFlatVectorsValues(
        int dimension,
        int size,
        IndexInput slice,
        DirectMonotonicReader ordToDocReader,
        VectorSimilarityFunction similarity
    ) {
        this.dimension = dimension;
        this.size = size;
        this.slice = slice;
        this.byteSize = dimension * Short.BYTES;
        this.value = new float[dimension];
        this.rawBytes = new byte[byteSize];
        this.ordToDocReader = ordToDocReader;
        this.similarity = similarity;
    }

    @Override
    public IndexInput getSlice() {
        return slice;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int getVectorByteLength() {
        return byteSize;
    }

    @Override
    public DocIndexIterator iterator() {
        return ordToDocReader == null ? createDenseIterator() : createSparseIterator();
    }

    @Override
    public int ordToDoc(int ord) {
        return ordToDocReader == null ? ord : (int) ordToDocReader.get(ord);
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        if (ord == lastOrd) {
            return value;
        }
        lastOrd = ord;
        readRawVectorBytes(ord, rawBytes, 0);
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(rawBytes, value, dimension, 0);
        return value;
    }

    /**
     * Reads the raw FP16 bytes for {@code ord} into {@code dest}, starting at
     * {@code destOffset}. Used by {@link KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer} to feed native SIMD
     * scoring without a Java-side FP16-to-FP32 decode.
     */
    void readRawVectorBytes(int ord, byte[] dest, int destOffset) throws IOException {
        slice.seek((long) ord * byteSize);
        slice.readBytes(dest, destOffset, byteSize);
    }

    int byteSize() {
        return byteSize;
    }

    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        if (size() == 0) {
            log.debug("No vectors present; returning null scorer from scorer(float[])");
            return null;
        }
        KNN1040HalfFloatFlatVectorsValues copy = copy();
        DocIndexIterator iterator = copy.iterator();
        RandomVectorScorer scorer = selectScorer(copy, target, similarity);
        log.debug("Selected scorer [{}] for similarity [{}]", scorer.getClass().getSimpleName(), similarity);
        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return scorer.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(scorer, iterator, matchingDocs);
            }
        };
    }

    @Override
    public KNN1040HalfFloatFlatVectorsValues copy() throws IOException {
        return new KNN1040HalfFloatFlatVectorsValues(dimension, size, slice.clone(), ordToDocReader, similarity);
    }

    /**
     * Extracts the mmap address for {@code values} (if available) and delegates the actual scorer
     * choice to {@link #selectNativeOrJavaScorer}. Shared by both the reader's
     * {@code getRandomVectorScorer} and this class's {@link #scorer(float[])} - both flat/exact-search
     * paths, never called during HNSW graph traversal (that goes through
     * {@link KNN1040HalfFloatVectorScorer#getRandomVectorScorerSupplier} instead).
     */
    static RandomVectorScorer selectScorer(KNN1040HalfFloatFlatVectorsValues values, float[] target, VectorSimilarityFunction similarity)
        throws IOException {
        long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
            values.getSlice(),
            0,
            values.getSlice().length()
        );
        return selectNativeOrJavaScorer(values, target, similarity, addressAndSize);
    }

    /**
     * Picks native SIMD or plain-Java scoring for {@code values} (never a {@code FlatVectorsScorer}
     * delegate chain - see this class's javadoc for why). See
     * {@link KNN1040HalfFloatVectorScorer#usesNativeScorer} for how native-vs-fallback is decided.
     * Shared with {@link KNN1040HalfFloatVectorScorer#getRandomVectorScorerSupplier} (HNSW graph
     * build/search) so the two entry points can't drift apart. Wrapped in
     * {@link PrefetchableRandomVectorScorer} either way, so this still prefetches ahead of bulk
     * scoring during graph traversal.
     */
    static RandomVectorScorer selectNativeOrJavaScorer(
        KNN1040HalfFloatFlatVectorsValues values,
        float[] target,
        VectorSimilarityFunction similarity,
        long[] addressAndSize
    ) {
        SimdVectorComputeService.SimilarityFunctionType nativeType = NativeEngines990KnnVectorsScorer.getNativeFunctionType(similarity);
        RandomVectorScorer.AbstractRandomVectorScorer scorer;
        if (KNN1040HalfFloatVectorScorer.usesNativeScorer(nativeType, addressAndSize)) {
            log.debug(
                "Selected native SIMD scorer for similarity [{}], nativeType [{}], mmap [{}]",
                similarity,
                nativeType,
                addressAndSize != null
            );
            scorer = new KNN1040HalfFloatVectorScorer.HalfFloatRandomVectorScorer(values, target, nativeType, addressAndSize);
        } else {
            log.debug(
                "Selected Java fallback scorer for similarity [{}]; nativeType [{}], simdSupported [{}]",
                similarity,
                nativeType,
                SimdFp16.isSIMDSupported()
            );
            scorer = new RandomVectorScorer.AbstractRandomVectorScorer(values) {
                @Override
                public float score(int node) throws IOException {
                    return similarity.compare(target, values.vectorValue(node));
                }
            };
        }
        return new PrefetchableRandomVectorScorer(scorer);
    }
}
