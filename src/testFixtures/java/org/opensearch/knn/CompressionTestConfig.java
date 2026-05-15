/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.HashMap;
import java.util.Map;

/**
 * Test configurations for dual-path coverage of FP32 (no compression) and SQ 1-bit (32x compression).
 * Provides compression-specific thresholds, eligibility flags, and helper methods for parameterized testing.
 */
@Getter
@AllArgsConstructor
public enum CompressionTestConfig {
    FP32(CompressionLevel.x1, Mode.IN_MEMORY, null, createFP32Thresholds(), true, true, true, true),
    SQ_1BIT(CompressionLevel.x32, Mode.ON_DISK, ScalarQuantizationType.ONE_BIT, createSQ1BitThresholds(), true, true, true, true);

    private final CompressionLevel compressionLevel;
    private final Mode mode;
    private final ScalarQuantizationType expectedQuantizationType;
    private final Map<SpaceType, Float> recallThresholds;
    private final boolean radialSearchEligible;
    private final boolean mosEligible;
    private final boolean scriptScoringEligible;
    private final boolean searchEligible;

    private static Map<SpaceType, Float> createFP32Thresholds() {
        Map<SpaceType, Float> thresholds = new HashMap<>();
        thresholds.put(SpaceType.L2, 0.95f);
        thresholds.put(SpaceType.COSINESIMIL, 0.95f);
        thresholds.put(SpaceType.INNER_PRODUCT, 0.95f);
        return thresholds;
    }

    private static Map<SpaceType, Float> createSQ1BitThresholds() {
        Map<SpaceType, Float> thresholds = new HashMap<>();
        thresholds.put(SpaceType.L2, 0.70f);
        thresholds.put(SpaceType.COSINESIMIL, 0.70f);
        thresholds.put(SpaceType.INNER_PRODUCT, 0.60f);
        return thresholds;
    }

    public String getCompressionLevelName() {
        return compressionLevel.getName();
    }

    public String getModeName() {
        return mode.getName();
    }

    public boolean isCompressed() {
        return this != FP32;
    }

    /**
     * Get minimum recall threshold for a specific space type.
     * Falls back to L2 threshold if space type not found.
     */
    public float getMinRecallThreshold(SpaceType spaceType) {
        return recallThresholds.getOrDefault(spaceType, recallThresholds.get(SpaceType.L2));
    }

    /**
     * Get minimum recall threshold using default L2 space type for backward compatibility.
     */
    public float getMinRecallThreshold() {
        return getMinRecallThreshold(SpaceType.L2);
    }

    /**
     * Check if this configuration is eligible for a specific test category.
     */
    public boolean isEligibleFor(TestCategory category) {
        switch (category) {
            case SEARCH:
                return searchEligible;
            case RADIAL:
                return radialSearchEligible;
            case MOS:
                return mosEligible;
            case SCRIPT_SCORING:
                return scriptScoringEligible;
            case INFRASTRUCTURE:
                return true; // All configs support infrastructure tests
            default:
                return false;
        }
    }

    /**
     * Test categories for eligibility checking.
     */
    public enum TestCategory {
        SEARCH,
        RADIAL,
        MOS,
        SCRIPT_SCORING,
        INFRASTRUCTURE
    }
}
