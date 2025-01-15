/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

@FunctionalInterface
public interface DocIdGrouper {
    int getGroupId(int childDocId);
}
