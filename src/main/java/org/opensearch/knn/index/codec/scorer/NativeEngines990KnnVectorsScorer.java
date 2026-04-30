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
        final SimdVectorComputeService.SimilarityFunctionType nativeType = getNativeFunctionType(similarityFunction);
        if (nativeType != null) {
            final FloatVectorValues bottomValues = WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues);
            if (bottomValues instanceof MMapVectorValues mmapValues) {
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
        final VectorSimilarityFunction similarityFunction
    ) {
        return switch (similarityFunction) {
            case MAXIMUM_INNER_PRODUCT -> SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT;
            case EUCLIDEAN -> SimdVectorComputeService.SimilarityFunctionType.FP16_L2;
            default -> null;
        };
    }
}
