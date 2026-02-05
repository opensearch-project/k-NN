/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import java.io.IOException;

/**
 * Exception thrown when native index building is aborted during merge operations.
 *
 * <p>This exception is typically thrown when a Lucene merge operation is cancelled
 * and the associated native index building process needs to be terminated gracefully.
 */
public class IndexBuildAbortedException extends IOException {

    /**
     * Constructs an IndexBuildAbortedException with the specified detail message.
     *
     * @param message the detail message explaining why the index build was aborted
     */
    public IndexBuildAbortedException(String message) {
        super(message);
    }

    /**
     * Constructs an IndexBuildAbortedException with the specified detail message and cause.
     *
     * @param message the detail message explaining why the index build was aborted
     * @param cause the underlying cause of the abort
     */
    public IndexBuildAbortedException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs an IndexBuildAbortedException with the specified cause.
     *
     * @param cause the underlying cause of the abort
     */
    public IndexBuildAbortedException(Throwable cause) {
        super(cause);
    }
}
