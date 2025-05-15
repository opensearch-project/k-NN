/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

/**
 * Provider for returning an instance of {@link FaissHNSW}.
 * <p>
 * This provider should consistently return the same instance across multiple calls.
 * It does not imply a singleton instance, but each provider instance should consistently return the same {@link FaissHNSW} instance across
 * multiple calls.
 */
public interface FaissHNSWProvider {
    /**
     * Return internal {@link FaissHNSW}.
     *
     * @return {@link FaissHNSW} instance it internally keeps.
     */
    FaissHNSW getFaissHnsw();
}
