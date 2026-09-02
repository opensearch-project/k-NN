/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

/**
 * The search side parameters of a {@link NativeEngineService} query. filterIdsType discriminates the
 * filteredIds encoding.
 *
 * <p>Construct through the per operation factory so only the fields that operation defines are set.
 * A field the factory does not take is zero and carries no meaning for that operation; implementations
 * must not read it: {@code k} exists only for {@code queryIndex}, {@code radius} and
 * {@code indexMaxResultWindow} only for {@code radiusQueryIndex}.
 */
@ExperimentalApi
public record NativeSearchParams(float[] queryVector, int k, float radius, int indexMaxResultWindow, EngineParameters methodParameters,
    long[] filteredIds, int filterIdsType, int[] parentIds) {

    /** Parameters of {@code queryIndex}: top-k search, {@code radius} and {@code indexMaxResultWindow} are not set. */
    public static NativeSearchParams forTopK(
        float[] queryVector,
        int k,
        EngineParameters methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        return new NativeSearchParams(queryVector, k, 0f, 0, methodParameters, filteredIds, filterIdsType, parentIds);
    }

    /** Parameters of {@code radiusQueryIndex}: radial search, {@code k} is not set. */
    public static NativeSearchParams forRadial(
        float[] queryVector,
        float radius,
        int indexMaxResultWindow,
        EngineParameters methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        return new NativeSearchParams(
            queryVector,
            0,
            radius,
            indexMaxResultWindow,
            methodParameters,
            filteredIds,
            filterIdsType,
            parentIds
        );
    }
}
