/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

/**
 * A specialized {@link Lucene104ScalarQuantizedVectorScorer} that leverages
 * FAISS-style SIMD-accelerated scoring for scalar-quantized vectors with fallback.
 * Will hit fallback only if we cannot use native bulk simd scorer.
 *
 * <p>This scorer attempts to use a native SIMD-backed bulk scoring path when:
 * <ul>
 *   <li>The underlying vector values are {@link QuantizedByteVectorValues}</li>
 *   <li>The backing storage can expose a raw memory address</li>
 *   <li>The scalar encoding matches the expected FAISS-compatible format</li>
 * </ul>
 *
 * <p>If these conditions are not met, it falls back to the default Lucene scoring implementation.
 *
 * <p>The SIMD path uses a precomputed search context and performs scoring in native code
 * (e.g., AVX-512), significantly improving throughput for large-scale vector search.
 *
 * <p>All scorers returned by {@link #getRandomVectorScorer} are wrapped with
 * {@link PrefetchableRandomVectorScorer} to prefetch vector data ahead of bulk scoring
 * operations, improving cache locality and reducing I/O latency during graph traversal.
 */
@Log4j2
public class KNN1040ScalarQuantizedVectorScorer extends Lucene104ScalarQuantizedVectorScorer {
    /**
     * Creates a new scorer that wraps a non-quantized delegate scorer.
     *
     * @param delegate fallback scorer used when SIMD acceleration is not applicable
     */
    public KNN1040ScalarQuantizedVectorScorer(final FlatVectorsScorer delegate) {
        super(delegate);
    }

    /**
     * Returns a {@link RandomVectorScorer} for the given query vector.
     *
     * <p><b>Important:</b> This method only supports {@link QuantizedByteVectorValues}. It will fail
     * with an exception if called with raw (non-quantized) vector values such as
     * {@code OffHeapFloatVectorValues}. Callers must ensure that this scorer is not used as the
     * scorer for raw vector formats (e.g., {@link org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat}).
     *
     * <p>This method attempts to construct a SIMD-accelerated scorer when the input vectors
     * are quantized and backed by memory that can be accessed directly (e.g., via a memory segment).
     * Otherwise, it falls back to the parent's quantized scoring implementation.
     *
     * @param similarityFunction the similarity function (e.g., inner product or L2)
     * @param vectorValues       the quantized vector storage (must be {@link QuantizedByteVectorValues})
     * @param target             the query vector (float32)
     * @return a scorer capable of computing similarity scores
     * @throws IOException if an error occurs while accessing vector data
     */
    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        // For the sparse case, KnnVectorValues having `QuantizedByteVectorValues` might be wrapped to support
        // vector ordinal to doc id mapping. For the dense case, it's not needed as vector ordinal is always the same
        // as doc id.
        if (vectorValues instanceof WrappedFloatVectorValues) {
            vectorValues = WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues);
        }

        final QuantizedByteVectorValues quantizedByteVectorValues;
        if (vectorValues instanceof QuantizedByteVectorValues) {
            quantizedByteVectorValues = (QuantizedByteVectorValues) vectorValues;
        } else {
            // Extract QuantizedByteVectorValues from `vectorValues`.
            // This should not be null, otherwise it can't get entroid + correction factors.
            quantizedByteVectorValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(vectorValues);
        }

        return new PrefetchableRandomVectorScorer(getScorer(similarityFunction, quantizedByteVectorValues, target));
    }

    private RandomVectorScorer.AbstractRandomVectorScorer getScorer(
        final VectorSimilarityFunction similarityFunction,
        final QuantizedByteVectorValues quantizedByteVectorValues,
        final float[] target
    ) throws IOException {
        final IndexInput indexInput = quantizedByteVectorValues.getSlice();
        final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(indexInput, 0, indexInput.length());
        if (addressAndSize != null) {
            // Try bulk SIMD
            return bulkSimdRandomVectorScorer(quantizedByteVectorValues, target, addressAndSize, similarityFunction);
        }

        // Fallback
        log.warn("Bulk SIMD for SQ is not supported, falling back to Lucene's random vector scorer");
        return (RandomVectorScorer.AbstractRandomVectorScorer) super.getRandomVectorScorer(
            similarityFunction,
            quantizedByteVectorValues,
            target
        );
    }

    /**
     * Builds a SIMD-accelerated scorer using quantized vectors and a precomputed query.
     *
     * <p>This method:
     * <ol>
     *   <li>Validates the scalar encoding format</li>
     *   <li>Quantizes the query vector into the same representation as stored vectors</li>
     *   <li>Applies transformations (e.g., nibble transposition) required for SIMD efficiency</li>
     *   <li>Initializes a native SIMD search context</li>
     * </ol>
     *
     * <p>The resulting scorer uses bulk SIMD instructions to compute similarity scores.
     *
     * @param quantizedByteVectorValues the quantized vector storage
     * @param target                    the query vector (float32)
     * @param addressAndSize            raw memory address and size of vector data
     * @param similarityFunction        similarity function to use
     * @return a SIMD-accelerated scorer
     * @throws IOException if quantization or initialization fails
     */
    private BulkSimdRandomVectorScorer bulkSimdRandomVectorScorer(
        final QuantizedByteVectorValues quantizedByteVectorValues,
        final float[] target,
        final long[] addressAndSize,
        final VectorSimilarityFunction similarityFunction
    ) throws IOException {
        // Check encoding type
        final Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding scalarEncoding = quantizedByteVectorValues.getScalarEncoding();

        // We only support 32x quantization with 4 bit query quantization for search.
        if (scalarEncoding != SINGLE_BIT_QUERY_NIBBLE) {
            throw new IllegalStateException(String.format("SQ only supports %s encoding.", SINGLE_BIT_QUERY_NIBBLE));
        }

        // Validate dimensionality
        FlatVectorsScorer.checkDimensions(target.length, quantizedByteVectorValues.dimension());

        // Transpose query vector if it needs to
        final OptimizedScalarQuantizer quantizer = quantizedByteVectorValues.getQuantizer();
        final byte[] scratch = new byte[scalarEncoding.getDiscreteDimensions(quantizedByteVectorValues.dimension())];
        final byte[] targetQuantized;
        if (scalarEncoding.isAsymmetric() == false) {
            targetQuantized = scratch;
        } else {
            // Asymmetric encoding requires packed representation
            targetQuantized = new byte[scalarEncoding.getQueryPackedLength(scratch.length)];
        }

        // We make a copy as the quantization process mutates the input
        final float[] targetCopy = ArrayUtil.copyOfSubArray(target, 0, target.length);

        // For cosine similarity, the query vector is expected to already be normalized.
        // Normalization is performed upfront in KNNQueryBuilder via VectorTransformerFactory
        // for Lucene cosine with SQ 1-bit and flat methods and for Faiss.

        // Perform scalar quantization
        final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms = quantizer.scalarQuantize(
            targetCopy,
            scratch,
            scalarEncoding.getQueryBits(),
            quantizedByteVectorValues.getCentroid()
        );

        // Transpose half-bytes (nibbles) for SIMD-friendly layout
        OptimizedScalarQuantizer.transposeHalfByte(scratch, targetQuantized);

        // Return Bulk SIMD scorer
        return new BulkSimdRandomVectorScorer(
            targetQuantized,
            targetCorrectiveTerms,
            addressAndSize,
            quantizedByteVectorValues,
            similarityFunction,
            targetCopy.length,
            quantizedByteVectorValues.getCentroidDP()
        );
    }

    /**
     * A {@link RandomVectorScorer} implementation backed by native SIMD computation.
     *
     * <p>This scorer delegates all similarity computations to a native service
     * ({@link SimdVectorComputeService}), which uses preloaded query state and raw
     * vector memory to compute scores efficiently using SIMD instructions.
     *
     * <p>The query is preprocessed (quantized + transformed) once during construction,
     * and reused across all scoring calls.
     */
    private static class BulkSimdRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        /**
         * Constructs a SIMD-backed scorer and initializes the native search context.
         *
         * <p>This constructor pushes all necessary query state into native memory,
         * including quantized query values and correction terms required for accurate scoring.
         *
         * @param targetQuantized       quantized query vector
         * @param targetCorrectiveTerms correction terms from quantization
         * @param addressAndSize        raw memory location of vector data
         * @param knnVectorValues       vector storage abstraction
         * @param similarityFunction    similarity function (IP or L2)
         * @param dimension             vector dimensionality
         * @param centroidDp            centroid dot-product correction
         */
        public BulkSimdRandomVectorScorer(
            final byte[] targetQuantized,
            final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms,
            final long[] addressAndSize,
            final QuantizedByteVectorValues knnVectorValues,
            final VectorSimilarityFunction similarityFunction,
            final int dimension,
            final float centroidDp
        ) {
            super(knnVectorValues);

            // Initialize native SIMD search context
            SimdVectorComputeService.saveSQSearchContext(
                targetQuantized,
                targetCorrectiveTerms.lowerInterval(),
                targetCorrectiveTerms.upperInterval(),
                targetCorrectiveTerms.additionalCorrection(),
                targetCorrectiveTerms.quantizedComponentSum(),
                addressAndSize,
                (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
                    || similarityFunction == VectorSimilarityFunction.COSINE
                    || similarityFunction == VectorSimilarityFunction.DOT_PRODUCT)
                        ? SimdVectorComputeService.SimilarityFunctionType.SQ_IP.ordinal()
                        : SimdVectorComputeService.SimilarityFunctionType.SQ_L2.ordinal(),
                dimension,
                centroidDp
            );
        }

        /**
         * Computes similarity scores for multiple vectors in bulk using native SIMD code.
         *
         * <p>This method is optimized for throughput and should be preferred when scoring
         * large batches of vectors.
         *
         * @param internalVectorIds vector ordinals to score
         * @param scores            output buffer for similarity scores
         * @param numVectors        number of vectors to process
         * @return implementation-defined value (typically unused aggregate)
         */
        @Override
        public float bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
            return SimdVectorComputeService.scoreSimilarityInBulk(internalVectorIds, scores, numVectors);
        }

        /**
         * Computes the similarity score for a single vector using native SIMD code.
         *
         * @param internalVectorId the internal vector ID to score
         * @return the computed similarity score
         * @throws IOException if the native scoring operation fails
         */
        @Override
        public float score(final int internalVectorId) {
            return SimdVectorComputeService.scoreSimilarity(internalVectorId);
        }
    }
}
