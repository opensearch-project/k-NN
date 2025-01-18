/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

public class IdAndDistance {
    public static final int INVALID_DOC_ID = -1;

    public int id;
    public float distance;

    public IdAndDistance() {}

    public IdAndDistance(int id, float distance) {
        this.id = id;
        this.distance = distance;
    }
}
