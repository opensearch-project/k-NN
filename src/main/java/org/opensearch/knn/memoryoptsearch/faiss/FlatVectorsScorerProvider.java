/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.util.List;

/**
 * Provides the appropriate {@link FlatVectorsScorer} for a given field by iterating a registered list of
 * {@link FlatVectorsScorerResolver}s. The first resolver whose {@link FlatVectorsScorerResolver#canResolve}
 * returns {@code true} is used. Falls back to {@code delegateScorer} if no resolver matches.
 *
 * <p>To add support for a new scorer, implement {@link FlatVectorsScorerResolver} as a static inner class
 * in {@link FlatVectorsScorerResolver} and register it in {@link #FLAT_VECTORS_SCORER_RESOLVER_LIST}.
 */
@UtilityClass
public class FlatVectorsScorerProvider {

    private static final List<FlatVectorsScorerResolver> FLAT_VECTORS_SCORER_RESOLVER_LIST = List.of(
        new FlatVectorsScorerResolver.AdcScorerResolver(),
        new FlatVectorsScorerResolver.FaissSQScorerResolver(),
        new FlatVectorsScorerResolver.HammingScorerResolver()
    );

    /**
     * Returns the appropriate {@link FlatVectorsScorer} for the given field.
     * Iterates registered {@link FlatVectorsScorerResolver}s in order; falls back to {@code delegateScorer}.
     *
     * @param fieldInfo           the field metadata containing space type and quantization attributes
     * @param similarityFunction  the similarity function for the query
     * @param delegateScorer      the default scorer to fall back to when no resolver matches
     * @return the resolved {@link FlatVectorsScorer}
     */
    public static FlatVectorsScorer getFlatVectorsScorer(
        final FieldInfo fieldInfo,
        final KNNVectorSimilarityFunction similarityFunction,
        final FlatVectorsScorer delegateScorer
    ) {
        return FLAT_VECTORS_SCORER_RESOLVER_LIST.stream()
            .filter(r -> r.canResolve(fieldInfo, similarityFunction))
            .map(r -> r.resolve(fieldInfo, similarityFunction, delegateScorer))
            .findFirst()
            .orElse(delegateScorer);
    }
}
