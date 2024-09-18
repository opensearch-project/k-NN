/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Builder
@EqualsAndHashCode
public final class RescoreContext {

    public static final float DEFAULT_OVERSAMPLE_FACTOR = 1.0f;
    public static final float MAX_OVERSAMPLE_FACTOR = 100.0f;
    public static final float MIN_OVERSAMPLE_FACTOR = 1.0f;

    public static final int MAX_FIRST_PASS_RESULTS = 10000;

    // Todo:- We will improve this in upcoming releases
    public static final int MIN_FIRST_PASS_RESULTS = 100;

    @Builder.Default
    private float oversampleFactor = DEFAULT_OVERSAMPLE_FACTOR;

    /**
     *
     * @return default RescoreContext
     */
    public static RescoreContext getDefault() {
        return RescoreContext.builder().build();
    }

    /**
     * Gets the number of results to return for the first pass of rescoring.
     *
     * @param finalK The final number of results to return for the entire shard
     * @return The number of results to return for the first pass of rescoring
     */
    public int getFirstPassK(int finalK) {
        return Math.min(MAX_FIRST_PASS_RESULTS, Math.max(MIN_FIRST_PASS_RESULTS, (int) Math.ceil(finalK * oversampleFactor)));
    }
}
