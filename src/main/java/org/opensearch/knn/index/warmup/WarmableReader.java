/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import java.io.IOException;

/**
 * A reader that supports explicit warm-up of its on-storage data structures.
 * <p>
 * Implementations are expected to force the underlying vector data (graphs, flat vectors,
 * quantized vectors, etc.) into the OS page cache so that the first real searches avoid
 * cold-read latency. This interface decouples the warm-up concern from the search path,
 * allowing {@link MemoryOptimizedSearchWarmup} to trigger warm-up without constructing
 * a fake search request.
 */
public interface WarmableReader {
    /**
     * Warms up all on-disk data structures associated with the given field.
     * <p>
     * After this method returns, subsequent searches on the field should not incur
     * cold-read I/O penalties because the relevant pages will already be in the OS
     * page cache.
     *
     * @param fieldName the name of the vector field to warm up
     * @throws IOException if an I/O error occurs while reading the underlying data
     */
    void warmUp(final String fieldName) throws IOException;
}
