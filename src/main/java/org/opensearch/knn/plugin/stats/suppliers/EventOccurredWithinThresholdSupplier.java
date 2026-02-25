/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.stats.suppliers;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.function.Supplier;

/**
 * Supplier that returns true if event occurred within the threshold, otherwise, false
 */
public class EventOccurredWithinThresholdSupplier implements Supplier<Boolean> {
    private final long threshold;
    private final ChronoUnit unit;
    private final Supplier<Instant> supplier;

    /**
     * Constructor
     *
     * @param supplier  which returns Instant
     * @param threshold duration which decides the threshold for given supplier
     * @param unit      determines the threshold's Time unit
     */
    public EventOccurredWithinThresholdSupplier(Supplier<Instant> supplier, long threshold, ChronoUnit unit) {
        this.supplier = supplier;
        this.threshold = threshold;
        this.unit = unit;
    }

    @Override
    public Boolean get() {

        Instant lastSeenAt = supplier.get();
        if (lastSeenAt == null)  // Event never happened
            return false;
        Instant expiringAt = lastSeenAt.plus(threshold, unit);
        // if expiration is greater than current instant, then event occurred
        if (expiringAt.compareTo(Instant.now()) > 0) {
            return true;
        }
        return false;
    }
}
