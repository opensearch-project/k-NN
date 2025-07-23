/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.Directory;

import java.io.IOException;

/**
 * Factory to create {@link VectorSearcher}.
 * Provided parameters will have {@link Directory} and a file name where implementation can rely on it to open an input stream.
 */
public interface VectorSearcherFactory {
    /**
     * Create a non-null {@link VectorSearcher} with given Lucene's {@link Directory}.
     *
     * @param directory Lucene's Directory.
     * @param fileName Logical file name to load.
     * @return Null instance if it is not supported, otherwise return {@link VectorSearcher}
     * @throws IOException
     */
    VectorSearcher createVectorSearcher(Directory directory, String fileName, boolean isAdc) throws IOException;
}
