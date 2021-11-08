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

import org.opensearch.knn.KNNTestCase;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.concurrent.TimeUnit;

public class EventOccurredWithinThresholdSupplierTests extends KNNTestCase {
    public void testOutsideThreshold() throws InterruptedException {
        Instant now = Instant.now();
        long threshold = 2;
        EventOccurredWithinThresholdSupplier supplier = new EventOccurredWithinThresholdSupplier(
            ()->now, threshold, ChronoUnit.SECONDS
        );
        TimeUnit.SECONDS.sleep(threshold + 1);
        assertFalse(supplier.get());
    }

    public void testEventNeverHappened() throws InterruptedException {
        long threshold = 2;
        EventOccurredWithinThresholdSupplier supplier = new EventOccurredWithinThresholdSupplier(
            () -> null, threshold, ChronoUnit.SECONDS
        );
        TimeUnit.SECONDS.sleep(threshold + 1);
        assertFalse(supplier.get());
    }

    public void testInsideThreshold() throws InterruptedException {
        Instant now = Instant.now();
        long threshold = 2;
        EventOccurredWithinThresholdSupplier supplier = new EventOccurredWithinThresholdSupplier(
            ()->now, threshold, ChronoUnit.MINUTES
        );
        TimeUnit.SECONDS.sleep(threshold + 1);
        assertTrue(supplier.get());
    }
}
