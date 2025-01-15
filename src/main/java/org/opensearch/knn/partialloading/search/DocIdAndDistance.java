/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

public class DocIdAndDistance {
    public static final int INVALID_DOC_ID = -1;

    public int id;
    public float distance;

    public DocIdAndDistance() {}

    public DocIdAndDistance(int id, float distance) {
        this.id = id;
        this.distance = distance;
    }
}
