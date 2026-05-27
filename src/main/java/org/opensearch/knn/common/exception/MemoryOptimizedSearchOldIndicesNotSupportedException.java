/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

/**
 * Exception thrown when memory-optimized search is attempted on indices created with older versions
 * that do not support this feature.
 */
public class MemoryOptimizedSearchOldIndicesNotSupportedException extends RuntimeException {

    /**
     * Constructs a new exception with the specified detail message.
     *
     * @param message the detail message describing which index or version is unsupported
     */
    public MemoryOptimizedSearchOldIndicesNotSupportedException(final String message) {
        super(message);
    }
}
