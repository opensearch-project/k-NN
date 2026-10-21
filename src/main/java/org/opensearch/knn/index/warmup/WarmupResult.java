/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.Getter;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

/**
 * Outcome of a k-NN warmup on a single shard. Carries the shard level skip reason (null when the
 * warmup was executed) along with the names of the k-NN fields that were not loaded, i.e. fields
 * whose engine does not create custom segment files and therefore have nothing to load off-heap.
 */
@Getter
public class WarmupResult {

    private final WarmupSkipReason skipReason;
    private final List<String> skippedFields;

    /**
     * Creates a result for a shard whose warmup was skipped.
     *
     * @param skipReason reason the warmup was skipped, must not be null
     * @param skippedFields names of the k-NN fields that were not loaded; empty when the skip is
     *        unrelated to fields
     */
    public static WarmupResult skipped(final WarmupSkipReason skipReason, final Collection<String> skippedFields) {
        return new WarmupResult(Objects.requireNonNull(skipReason, "skipReason must not be null"), skippedFields);
    }

    /**
     * Creates a result for a shard whose warmup was executed. Individual fields may still have
     * been skipped.
     *
     * @param skippedFields names of the k-NN fields that were not loaded; may be empty
     */
    public static WarmupResult executed(final Collection<String> skippedFields) {
        return new WarmupResult(null, skippedFields);
    }

    private WarmupResult(final WarmupSkipReason skipReason, final Collection<String> skippedFields) {
        this.skipReason = skipReason;
        this.skippedFields = List.copyOf(skippedFields);
    }
}
