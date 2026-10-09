/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.Getter;

/**
 * Reasons why a k-NN warmup was skipped for a shard. When warmup is skipped, the warmup API still
 * succeeds, but no data is actually loaded.
 */
@Getter
public enum WarmupSkipReason {
    /**
     * The shard belongs to a warm-tier index.
     */
    WARM_TIER_INDEX("warm_tier_index"),

    /**
     * No k-NN field of the shard uses an engine that creates custom segment files, so warmup has
     * nothing to load off-heap. Currently this means all fields use the Lucene engine.
     */
    LUCENE_ENGINE("lucene_engine"),

    /**
     * The skip reason could not be recognized.
     */
    UNKNOWN("unknown");

    private final String value;

    WarmupSkipReason(final String value) {
        this.value = value;
    }
}
