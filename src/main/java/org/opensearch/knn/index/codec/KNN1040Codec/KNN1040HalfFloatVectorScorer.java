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

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.BULK_SCORE_BATCH_SIZE;

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
        // The unwrap above stops at MMapFloatVectorValues; unwrap one more layer and keep its address
        // for zero-copy scoring straight off mapped memory.
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
            // No native kernel or SIMD unavailable: never hand halfFloatValues to delegate, which
            // assumes 4 bytes/dimension and would overread this FP16 (2-byte) data.
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
     * node instead of one per graph edge.
     */
    private static final class HalfFloatRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        // Used for both raw-byte candidate reads (readRawVectorBytes) and decoding the "current"
        // graph node into a float[] target (vectorValue, in setScoringOrdinal below) - safe to share
        // since both always seek explicitly before reading, and only the target side ever decodes.
        private final KNN1040HalfFloatFlatVectorsValues vectorValues;
        private final SimdVectorComputeService.SimilarityFunctionType nativeType;
        private final long[] addressAndSize;

        HalfFloatRandomVectorScorerSupplier(
            KNN1040HalfFloatFlatVectorsValues vectorValues,
            SimdVectorComputeService.SimilarityFunctionType nativeType,
            long[] addressAndSize
        ) {
            this.vectorValues = vectorValues;
            this.nativeType = nativeType;
            this.addressAndSize = addressAndSize;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() {
            return new UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer(vectorValues) {
                private HalfFloatRandomVectorScorer delegate;
                private PrefetchableRandomVectorScorer prefetchableDelegate;

                @Override
                public void setScoringOrdinal(int node) throws IOException {
                    float[] target = vectorValues.vectorValue(node);
                    if (delegate == null) {
                        delegate = new HalfFloatRandomVectorScorer(vectorValues, target, nativeType, addressAndSize);
                        prefetchableDelegate = new PrefetchableRandomVectorScorer(delegate);
                    } else {
                        // Reuse the existing scorer/buffer instead of allocating a fresh one for every graph node
                        delegate.setTarget(target);
                    }
                }

                /**
                 * Scores {@code node} against whatever target {@link #setScoringOrdinal} last set,
                 * via the byte-based {@link HalfFloatRandomVectorScorer} built above - no decode here.
                 */
                @Override
                public float score(int node) throws IOException {
                    return requireDelegate().score(node);
                }

                /**
                 * Scores {@code numNodes} candidates from {@code nodes} against the current target in
                 * one call, returning the maximum score. {@code requireDelegate()} returns the
                 * {@link PrefetchableRandomVectorScorer} wrapper, so this also issues an I/O prefetch
                 * for the candidates' backing bytes before the real scoring happens - see the
                 * prefetch note on {@link #setScoringOrdinal} above.
                 */
                @Override
                public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
                    return requireDelegate().bulkScore(nodes, scores, numNodes);
                }

                private PrefetchableRandomVectorScorer requireDelegate() {
                    if (prefetchableDelegate == null) {
                        throw new IllegalStateException("setScoringOrdinal must be called before scoring");
                    }
                    return prefetchableDelegate;
                }
            };
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new HalfFloatRandomVectorScorerSupplier(vectorValues.copy(), nativeType, addressAndSize);
        }
    }

    /**
     * Scores FP16 vectors via native SIMD: directly off mapped memory when {@code addressAndSize} is
     * present (mmap-backed, zero-copy), otherwise by reading candidates into a heap buffer first.
     * {@link #setTarget} re-saves the native search context on every
     * {@link UpdateableRandomVectorScorer#setScoringOrdinal} call, since HNSW graph build scores
     * against a new "current" node each time, unlike search's fixed query.
     */
    static class HalfFloatRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final KNN1040HalfFloatFlatVectorsValues values;
        private final SimdVectorComputeService.SimilarityFunctionType nativeFunctionType;
        private final long[] addressAndSize;
        private final boolean usesMmapAddress;
        private byte[] vectorBytesBuffer;
        private final float[] singleScoreBuffer = new float[1];
        private final int[] singleVectorId = new int[] { 0 };
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
            this.usesMmapAddress = addressAndSize != null && addressAndSize.length > 0;
            if (!usesMmapAddress) {
                this.vectorBytesBuffer = new byte[values.byteSize() * BULK_SCORE_BATCH_SIZE];
            }
            setTarget(target);
        }

        // Repoints the native search context at a new target vector, reusing this scorer's buffers -
        // lets callers that score against many targets in sequence (e.g. HNSW graph build) avoid
        // allocating a fresh scorer per target.
        void setTarget(float[] target) {
            SimdVectorComputeService.saveSearchContext(
                target,
                usesMmapAddress ? addressAndSize : new long[0],
                nativeFunctionType.ordinal()
            );
        }

        @Override
        public float score(int node) throws IOException {
            if (usesMmapAddress) {
                return SimdVectorComputeService.scoreSimilarity(node);
            }
            values.readRawVectorBytes(node, vectorBytesBuffer, 0);
            SimdVectorComputeService.scoreSimilarityInBulkFromFp16Bytes(vectorBytesBuffer, 1, singleVectorId, singleScoreBuffer);
            return singleScoreBuffer[0];
        }

        @Override
        public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
            if (usesMmapAddress) {
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
