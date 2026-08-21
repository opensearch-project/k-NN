/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.jni.SimdFp16;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;

/**
 * Wraps a {@link FlatVectorsScorer} to give HNSW graph construction (flush's incremental build and
 * merge's graph rebuild) a decode-free scorer supplier for FP16 values, matching the decode-free path
 * search already gets via {@link KNN1040HalfFloatFlatVectorsValues#selectFallbackScorer}. Without
 * this, {@code getRandomVectorScorerSupplier} has no SIMD-aware path and decodes on every graph-edge
 * comparison instead of once per vector.
 *
 * <p>Delegates everything else unchanged, so it's safe to wrap any existing scorer chain - this class
 * only touches the one method, and only activates for our own {@link KNN1040HalfFloatFlatVectorsValues}.
 */
public class KNN1040HalfFloatVectorScorer implements FlatVectorsScorer {
    private final FlatVectorsScorer delegate;

    public KNN1040HalfFloatVectorScorer(FlatVectorsScorer delegate) {
        this.delegate = delegate;
    }

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) throws IOException {
        FloatVectorValues bottomValues = WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues);
        // MMapFloatVectorValues doesn't extend WrappedFloatVectorValues, so the unwrap above stops at
        // it rather than reaching the FP16 values it wraps - unwrap that one extra layer here. Keep its
        // address around: when present, candidate scoring can read straight from mapped memory instead
        // of copying each candidate's bytes out of the IndexInput slice first.
        long[] addressAndSize = null;
        if (bottomValues instanceof MMapFloatVectorValues mmapValues) {
            addressAndSize = mmapValues.getAddressAndSize();
            bottomValues = mmapValues.getDelegate();
        }
        if (bottomValues instanceof KNN1040HalfFloatFlatVectorsValues halfFloatValues) {
            final SimdVectorComputeService.SimilarityFunctionType nativeType = NativeEngines990KnnVectorsScorer.getNativeFunctionType(
                similarityFunction
            );
            if (nativeType != null && SimdFp16.isSIMDSupported()) {
                return new HalfFloatRandomVectorScorerSupplier(halfFloatValues, nativeType, addressAndSize);
            }
            // No native FP16 kernel for this similarity function (e.g. COSINE - see
            // NativeEngines990KnnVectorsScorer#getNativeFunctionType), or SIMD isn't available. Must
            // still never hand `halfFloatValues` to `delegate` below: that's Lucene's *optimized*
            // scorer chain, which detects HasIndexSlice and reads the raw slice assuming 4
            // bytes/dimension (float32), corrupting/crashing on this FP16 (2 bytes/dimension) data -
            // the same danger `KNN1040HalfFloatFlatVectorsValues#selectFallbackScorer` already avoids
            // for search. DefaultFlatVectorScorer is Lucene's plain, non-accelerated scorer: it only
            // ever calls FloatVectorValues#vectorValue(ord) - our own correct FP16 decode - and never
            // does any HasIndexSlice/memory-segment detection, so it's safe here even though the
            // values also implement HasIndexSlice for the (separate, guarded) mmap-native path above.
            return DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(similarityFunction, halfFloatValues);
        }
        return delegate.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        return delegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        return delegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
    }

    @Override
    public String toString() {
        return "KNN1040HalfFloatVectorScorer(delegate=" + delegate + ")";
    }

    /**
     * Builds {@link UpdateableRandomVectorScorer}s for HNSW graph construction that read raw FP16
     * bytes directly for candidate comparisons via {@link HalfFloatRandomVectorScorer} - the
     * same decode-free path search already uses. The "current" graph node set via
     * {@link UpdateableRandomVectorScorer#setScoringOrdinal} is decoded once (via
     * {@link KNN1040HalfFloatFlatVectorsValues#vectorValue}) to build the native search context;
     * every subsequent candidate comparison against it stays fully byte-based - one decode per graph
     * node instead of one per graph edge, versus the generic Lucene fallback this replaces.
     */
    private static final class HalfFloatRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        private final KNN1040HalfFloatFlatVectorsValues values;
        private final KNN1040HalfFloatFlatVectorsValues targetValues;
        private final SimdVectorComputeService.SimilarityFunctionType nativeType;
        private final long[] addressAndSize;

        HalfFloatRandomVectorScorerSupplier(
            KNN1040HalfFloatFlatVectorsValues values,
            SimdVectorComputeService.SimilarityFunctionType nativeType,
            long[] addressAndSize
        ) throws IOException {
            this.values = values;
            this.targetValues = values.copy();
            this.nativeType = nativeType;
            this.addressAndSize = addressAndSize;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() {
            return new UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer(values) {
                private HalfFloatRandomVectorScorer delegate;
                // Wraps `delegate` once it exists, prefetching candidate bytes ahead of each bulkScore
                // call - same PrefetchableRandomVectorScorer used for search's mmap tier and the flat
                // fallback tier. Built once alongside `delegate` and reused across setScoringOrdinal
                // calls, same as the buffer/native-context reuse below.
                private PrefetchableRandomVectorScorer prefetchableDelegate;

                @Override
                public void setScoringOrdinal(int node) throws IOException {
                    float[] target = targetValues.vectorValue(node);
                    if (delegate == null) {
                        delegate = new HalfFloatRandomVectorScorer(values, target, nativeType, addressAndSize);
                        prefetchableDelegate = new PrefetchableRandomVectorScorer(delegate);
                    } else {
                        // Reuse the existing scorer/buffer instead of allocating a fresh one for every
                        // graph node - only the native search context needs to change.
                        delegate.setTarget(target);
                    }
                }

                @Override
                public float score(int node) throws IOException {
                    if (prefetchableDelegate == null) {
                        throw new IllegalStateException("setScoringOrdinal must be called before score");
                    }
                    return prefetchableDelegate.score(node);
                }

                @Override
                public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
                    if (prefetchableDelegate == null) {
                        throw new IllegalStateException("setScoringOrdinal must be called before bulkScore");
                    }
                    return prefetchableDelegate.bulkScore(nodes, scores, numNodes);
                }
            };
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new HalfFloatRandomVectorScorerSupplier(values.copy(), nativeType, addressAndSize);
        }
    }

    /**
     * Scores FP16 vectors via native SIMD. When {@code addressAndSize} is available (the segment
     * being read from is mmap-backed), candidates are scored directly off mapped memory by ordinal -
     * the same zero-copy mechanism {@link org.opensearch.knn.memoryoptsearch.faiss.NativeRandomVectorScorer}
     * uses for search - via {@link SimdVectorComputeService#scoreSimilarity}/{@code scoreSimilarityInBulk}.
     * Otherwise candidates are read one at a time from the segment's {@link org.apache.lucene.store.IndexInput}
     * slice into a heap buffer before scoring via {@link SimdVectorComputeService#scoreSimilarityInBulkFromFp16Bytes}.
     *
     * The search context (query buffer + similarity function, plus the mmap address when present) is
     * saved once per target in {@link #setTarget}, since the "current" graph node being scored against
     * changes on every {@link UpdateableRandomVectorScorer#setScoringOrdinal} call during HNSW graph
     * build - unlike search's fixed-for-the-scorer's-lifetime query.
     */
    static class HalfFloatRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        // Matches KNN1040HalfFloatFlatVectorsReader's bulk batch size, so the non-mmap buffer is
        // pre-sized for a typical bulkScore() call instead of reallocating on the first one.
        private static final int BULK_SCORE_BATCH_SIZE = 64;
        private final KNN1040HalfFloatFlatVectorsValues values;
        private final SimdVectorComputeService.SimilarityFunctionType nativeFunctionType;
        private final long[] addressAndSize;
        private byte[] vectorBytesBuffer;
        private final float[] singleScoreBuffer = new float[1];
        private final int[] singleVectorId = new int[] { 0 };
        // Positional ids of the vectors packed into vectorBytesBuffer - only used without an mmap address
        private int[] identityIds = new int[0];

        HalfFloatRandomVectorScorer(
            KNN1040HalfFloatFlatVectorsValues values,
            float[] target,
            SimdVectorComputeService.SimilarityFunctionType nativeFunctionType,
            long[] addressAndSize
        ) {
            super(values);
            this.values = values;
            this.nativeFunctionType = nativeFunctionType;
            this.addressAndSize = addressAndSize;
            if (!usesMmapAddress()) {
                this.vectorBytesBuffer = new byte[values.byteSize() * BULK_SCORE_BATCH_SIZE];
            }
            setTarget(target);
        }

        private boolean usesMmapAddress() {
            return addressAndSize != null && addressAndSize.length > 0;
        }

        // Repoints the native search context at a new target vector, reusing this scorer's buffers -
        // lets callers that score against many targets in sequence (e.g. HNSW graph build) avoid
        // allocating a fresh scorer per target.
        void setTarget(float[] target) {
            SimdVectorComputeService.saveSearchContext(
                target,
                usesMmapAddress() ? addressAndSize : new long[0],
                nativeFunctionType.ordinal()
            );
        }

        @Override
        public float score(int node) throws IOException {
            if (usesMmapAddress()) {
                return SimdVectorComputeService.scoreSimilarity(node);
            }
            values.readRawVectorBytes(node, vectorBytesBuffer, 0);
            SimdVectorComputeService.scoreSimilarityInBulkFromFp16Bytes(vectorBytesBuffer, 1, singleVectorId, singleScoreBuffer);
            return singleScoreBuffer[0];
        }

        @Override
        public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
            if (usesMmapAddress()) {
                return SimdVectorComputeService.scoreSimilarityInBulk(nodes, scores, numNodes);
            }
            int byteSize = values.byteSize();
            int requiredBytes = numNodes * byteSize;
            if (vectorBytesBuffer.length < requiredBytes) {
                vectorBytesBuffer = new byte[requiredBytes];
            }
            for (int i = 0; i < numNodes; i++) {
                values.readRawVectorBytes(nodes[i], vectorBytesBuffer, i * byteSize);
            }
            growIdentityIds(numNodes);
            return SimdVectorComputeService.scoreSimilarityInBulkFromFp16Bytes(vectorBytesBuffer, numNodes, identityIds, scores);
        }

        private void growIdentityIds(int numNodes) {
            int previousLength = identityIds.length;
            if (previousLength >= numNodes) {
                return;
            }
            identityIds = new int[numNodes];
            for (int i = 0; i < numNodes; i++) {
                identityIds[i] = i;
            }
        }
    }
}
