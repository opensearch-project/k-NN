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
    public static final int DIMENSION_THRESHOLD = 1000;
    public static final float OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD = 5.0f;
    public static final float OVERSAMPLE_FACTOR_ABOVE_DIMENSION_THRESHOLD = 3.0f;

    // Dimension thresholds for adjusting oversample factor
    public static final int DIMENSION_THRESHOLD_1000 = 1000;
    public static final int DIMENSION_THRESHOLD_768 = 768;

    // Oversample factors based on dimension thresholds
    public static final float OVERSAMPLE_FACTOR_1000 = 1.0f;  // No oversampling for dimensions >= 1000
    public static final float OVERSAMPLE_FACTOR_768 = 2.0f;   // 2x oversampling for dimensions >= 768 and < 1000
    public static final float OVERSAMPLE_FACTOR_BELOW_768 = 3.0f; // 3x oversampling for dimensions < 768

    // Todo:- We will improve this in upcoming releases
    public static final int MIN_FIRST_PASS_RESULTS = 100;

    @Builder.Default
    private float oversampleFactor = DEFAULT_OVERSAMPLE_FACTOR;

    /**
     * Flag to track whether the oversample factor is user-provided or default. The Reason to introduce
     * this is to set default when Shard Level rescoring is false,
     * else we end up overriding user provided value in NativeEngineKnnVectorQuery
     */
    @Builder.Default
    private boolean userProvided = true;

    /**
     * Flag to track whether rescoring has been disabled by the query parameters.
     */
    @Builder.Default
    private boolean rescoreEnabled = true;

    public static final RescoreContext EXPLICITLY_DISABLED_RESCORE_CONTEXT = RescoreContext.builder()
        .oversampleFactor(DEFAULT_OVERSAMPLE_FACTOR)
        .rescoreEnabled(false)
        .build();

    /**
     *
     * @return default RescoreContext
     */
    public static RescoreContext getDefault() {
        return RescoreContext.builder().oversampleFactor(DEFAULT_OVERSAMPLE_FACTOR).userProvided(false).build();
    }

    /**
     * Calculates the number of results to return for the first pass of rescoring (firstPassK).
     * This method considers whether shard-level rescoring is enabled and adjusts the oversample factor
     * based on the vector dimension if shard-level rescoring is disabled.
     *
     * @param finalK The final number of results to return for the entire shard.
     * @param isShardLevelRescoringDisabled A boolean flag indicating whether shard-level rescoring is disabled.
     *                                     If false, the dimension-based oversampling logic is bypassed.
     * @param dimension The dimension of the vector. This is used to determine the oversampling factor when
     *                  shard-level rescoring is disabled.
     * @return The number of results to return for the first pass of rescoring, adjusted by the oversample factor.
     */
    public int getFirstPassK(int finalK, boolean isShardLevelRescoringDisabled, int dimension) {
        // Only apply default dimension-based oversampling logic when:
        // 1. Shard-level rescoring is disabled
        // 2. The oversample factor was not provided by the user
        if (isShardLevelRescoringDisabled && !userProvided) {
            // Apply new dimension-based oversampling logic when shard-level rescoring is disabled
            if (dimension >= DIMENSION_THRESHOLD_1000) {
                oversampleFactor = OVERSAMPLE_FACTOR_1000;  // No oversampling for dimensions >= 1000
            } else if (dimension >= DIMENSION_THRESHOLD_768) {
                oversampleFactor = OVERSAMPLE_FACTOR_768;   // 2x oversampling for dimensions >= 768 and < 1000
            } else {
                oversampleFactor = OVERSAMPLE_FACTOR_BELOW_768;  // 3x oversampling for dimensions < 768
            }
        }
        // The calculation for firstPassK remains the same, applying the oversample factor
        return Math.min(MAX_FIRST_PASS_RESULTS, Math.max(MIN_FIRST_PASS_RESULTS, (int) Math.ceil(finalK * oversampleFactor)));
    }

}
