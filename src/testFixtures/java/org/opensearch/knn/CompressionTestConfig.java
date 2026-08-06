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

import java.util.HashMap;
import java.util.Map;

/**
 * Test configurations for dual-path coverage of an unconfigured index (no explicit compression) and 32x compression.
 * Provides compression-specific thresholds, eligibility flags, and helper methods for parameterized testing.
 */
@Getter
@AllArgsConstructor
public enum CompressionTestConfig {
    X1(CompressionLevel.x1, Mode.NOT_CONFIGURED, createDefaultThresholds(), true, true, true, true),
    X32(CompressionLevel.x32, Mode.ON_DISK, createX32Thresholds(), true, true, true, true);

    private final CompressionLevel compressionLevel;
    private final Mode mode;
    private final Map<SpaceType, Float> recallThresholds;
    private final boolean radialSearchEligible;
    private final boolean mosEligible;
    private final boolean scriptScoringEligible;
    private final boolean searchEligible;

    private static Map<SpaceType, Float> createDefaultThresholds() {
        Map<SpaceType, Float> thresholds = new HashMap<>();
        thresholds.put(SpaceType.L2, 0.95f);
        thresholds.put(SpaceType.COSINESIMIL, 0.95f);
        thresholds.put(SpaceType.INNER_PRODUCT, 0.95f);
        return thresholds;
    }

    private static Map<SpaceType, Float> createX32Thresholds() {
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
        return this != X1;
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
