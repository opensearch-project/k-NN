/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

/**
 * This determines whether the provided doc id should be included in the results.
 */
public interface MatchDocSelector {
    /**
     * Returns True if `docId` should be included in the results, otherwise False.
     *
     * @param docId Document id.
     * @return True if `docId` should be included in the results, otherwise False.
     */
    boolean test(int docId);
}
