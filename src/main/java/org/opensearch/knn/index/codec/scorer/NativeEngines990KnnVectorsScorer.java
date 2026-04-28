/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.MMapVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.NativeRandomVectorScorer;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;

/**
 * A {@link FlatVectorsScorer} that transparently selects between native SIMD-optimized scoring
 * and pure-Java scoring based on whether the underlying vector values are memory-mapped.
 *
 * <p>When the bottom-level {@link FloatVectorValues} implements {@link MMapVectorValues},
 * this scorer returns a {@link NativeRandomVectorScorer} for hardware-accelerated computation.
 * Otherwise, it delegates to the wrapped {@link FlatVectorsScorer}.
 *
 * <p>Supports both FP16 and BF16 scalar quantized indices, selecting the appropriate
 * native similarity function type based on the underlying quantizer.
 */
@RequiredArgsConstructor
public class NativeEngines990KnnVectorsScorer implements FlatVectorsScorer {
    private final FlatVectorsScorer delegate;

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        final FloatVectorValues bottomValues = WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues);
        if (bottomValues instanceof MMapVectorValues mmapValues) {
            final boolean isBF16 = (bottomValues instanceof MMapFloatVectorValues mmapFloatValues)
                && mmapFloatValues.getQuantizerType() == FaissQuantizerType.QT_BF16;
            final SimdVectorComputeService.SimilarityFunctionType nativeType = getNativeFunctionType(similarityFunction, isBF16);
            if (nativeType != null) {
                return new NativeRandomVectorScorer(target, vectorValues, mmapValues, nativeType);
            }
        }
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
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) throws IOException {
        return delegate.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
    }

    private static SimdVectorComputeService.SimilarityFunctionType getNativeFunctionType(
        final VectorSimilarityFunction similarityFunction,
        final boolean isBF16
    ) {
        return switch (similarityFunction) {
            case MAXIMUM_INNER_PRODUCT -> isBF16
                ? SimdVectorComputeService.SimilarityFunctionType.BF16_MAXIMUM_INNER_PRODUCT
                : SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT;
            case EUCLIDEAN -> isBF16
                ? SimdVectorComputeService.SimilarityFunctionType.BF16_L2
                : SimdVectorComputeService.SimilarityFunctionType.FP16_L2;
            default -> null;
        };
    }
}
