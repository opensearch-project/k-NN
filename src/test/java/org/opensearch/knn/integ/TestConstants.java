/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Constants for mixed vector document testing.
 */
public final class TestConstants {

    // Response patterns
    public static final String HITS_PATTERN = "\"hits\":";
    public static final String NON_VECTOR_DOC_ID = "\"10\"";
    public static final String COUNT_10 = "\"count\":10";
    public static final String COUNT_9 = "\"count\":9";
    public static final String VECTORIZED = "vectorized";
    public static final String NON_VECTORIZED = "non-vectorized";

    // Common messages
    public static final String SEARCH_SUCCESS_MSG = "Should return search results for ";
    public static final String NON_VECTOR_EXCLUDE_MSG = "Non-vector document should not appear in k-NN results for ";
    public static final String FILTERED_SUCCESS_MSG = "Filtered search should return results for ";
    public static final String SCRIPT_SUCCESS_MSG = "Script search should return results for ";
    public static final String TOTAL_DOCS_MSG = "Should have 10 total documents for ";
    public static final String VECTOR_DOCS_MSG = "Should have 9 documents with vectors for ";

    // Engine arrays
    public static final KNNEngine[] TEST_ENGINES = { KNNEngine.FAISS, KNNEngine.LUCENE };

    private TestConstants() {} // Utility class
}
