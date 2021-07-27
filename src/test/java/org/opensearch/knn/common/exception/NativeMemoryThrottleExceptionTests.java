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

package org.opensearch.knn.common.exception;

import org.opensearch.knn.KNNTestCase;

public class NativeMemoryThrottleExceptionTests extends KNNTestCase {
    public void testException() {
        String message = "test message";
        NativeMemoryThrottleException nativeMemoryThrottleException = new NativeMemoryThrottleException(message);
        assertEquals(message, nativeMemoryThrottleException.getMessage());
    }
}
