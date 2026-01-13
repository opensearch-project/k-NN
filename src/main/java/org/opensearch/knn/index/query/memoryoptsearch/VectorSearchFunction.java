/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;

import java.io.IOException;

/**
 * Functional interface for vector search operations.
 * Represents a search function that can be invoked with field name, vector, collector, and accept docs.
 */
@FunctionalInterface
public interface VectorSearchFunction {
    /**
     * Performs a vector search operation.
     *
     * @param fieldName the name of the vector field
     * @param vector the query vector (either float[] or byte[])
     * @param knnCollector the collector for gathering results
     * @param acceptDocs the document filter
     * @throws IOException if an I/O error occurs during search
     */
    void search(String fieldName, Object vector, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException;
}
