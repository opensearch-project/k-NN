/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

public class KNNResult {
    private String docId;
    private Float[] vector;

    public KNNResult(String docId, Float[] vector) {
        this.docId = docId;
        this.vector = vector;
    }

    public String getDocId() {
        return docId;
    }

    public  Float[] getVector() {
        return vector;
    }
}