/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

public class UnsupportedFaissIndexException extends RuntimeException {
    public UnsupportedFaissIndexException(final String message) {
        super(message);
    }
}
