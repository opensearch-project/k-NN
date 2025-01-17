/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

/**
 * This grouper returns a group id that a given child document id belongs to.
 */
@FunctionalInterface
public interface DocIdGrouper {
    int getGroupId(int childDocId);
}
