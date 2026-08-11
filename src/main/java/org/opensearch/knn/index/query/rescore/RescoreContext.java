/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@Builder
@EqualsAndHashCode
public final class RescoreContext {
    public static final int NO_RESCORE_NEEDED = 0;

    public static final float FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR = 2.0f;

    public static final float DEFAULT_OVERSAMPLE_FACTOR = 1.0f;
    public static final float MAX_OVERSAMPLE_FACTOR = 100.0f;
    public static final float MIN_OVERSAMPLE_FACTOR = 1.0f;

    public static final int MAX_FIRST_PASS_RESULTS = 10000;
    public static final int DIMENSION_THRESHOLD = 1000;

    public static final float OVERSAMPLE_FACTOR_DEFAULT_FOR_LUCENE_SCALAR_QUANTIZER_AFTER_V360 = 2.0f;
    public static final float OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD = 5.0f;

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
    private final float oversampleFactor = DEFAULT_OVERSAMPLE_FACTOR;

    /**
     * Flag to track whether the oversample factor is user-provided or default. The Reason to introduce
     * this is to set default when Shard Level rescoring is false,
     * else we end up overriding user provided value in NativeEngineKnnVectorQuery
     */
    @Builder.Default
    private final boolean userProvided = true;

    /**
     * Flag to track whether rescoring has been disabled by the query parameters.
     */
    @Builder.Default
    private final boolean rescoreEnabled = true;

    /**
     * Flag to control whether the dimension-based oversampling logic in {@link #getFirstPassK(int, boolean, int)}
     * is allowed to override the configured oversample factor. When set to {@code false}, the oversample factor
     * remains fixed regardless of vector dimension. This is used for encoders like FAISS SQ where the oversample
     * factor should not be adjusted based on dimension.
     */
    @Builder.Default
    private final boolean allowOverrideOversampleFactor = true;

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
     * Adjusts the oversample factor based on vector dimension when the factor was not explicitly
     * provided by the user and the encoder allows override. Higher dimensions need less oversampling
     * because quantization error is relatively smaller per-dimension.
     *
     * @param finalK The final number of results to return for the entire shard.
     * @param dimension The dimension of the vector. Used to determine the appropriate oversampling factor.
     * @return The number of results to return for the first pass of rescoring, adjusted by the oversample factor.
     */
    public int getFirstPassK(int finalK, int dimension) {
        float effectiveOversampleFactor = oversampleFactor;
        // Apply dimension-based oversampling when the oversample factor was not user-provided
        // and the encoder allows override. This ensures high-dimensional vectors (which have
        // lower relative quantization error) don't over-fetch candidates unnecessarily.
        if (!userProvided && allowOverrideOversampleFactor) {
            if (dimension >= DIMENSION_THRESHOLD_1000) {
                effectiveOversampleFactor = OVERSAMPLE_FACTOR_1000;
            } else if (dimension >= DIMENSION_THRESHOLD_768) {
                effectiveOversampleFactor = OVERSAMPLE_FACTOR_768;
            } else {
                effectiveOversampleFactor = OVERSAMPLE_FACTOR_BELOW_768;
            }
        }
        return Math.min(MAX_FIRST_PASS_RESULTS, Math.max(MIN_FIRST_PASS_RESULTS, (int) Math.ceil(finalK * effectiveOversampleFactor)));
    }

    /**
     * @deprecated Use {@link #getFirstPassK(int, int)} instead. The {@code isShardLevelRescoringDisabled} parameter
     * is no longer used — dimension-based oversampling now applies unconditionally when the oversample factor is not
     * user-provided. This method will be removed in the next major version.
     */
    @Deprecated
    public int getFirstPassK(int finalK, boolean isShardLevelRescoringDisabled, int dimension) {
        return getFirstPassK(finalK, dimension);
    }

}
