/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import java.util.Arrays;
import java.util.Objects;

public class KNNResult {
    private final static float delta = 1e-3f;

    private String docId;
    private float[] vector;
    private Float score;

    public KNNResult(String docId, float[] vector, Float score) {
        this.docId = docId;
        this.vector = vector;
        this.score = score;
    }

    public String getDocId() {
        return docId;
    }

    public float[] getVector() {
        return vector;
    }

    public Float getScore() {
        return score;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KNNResult knnResult = (KNNResult) o;
        return Objects.equals(docId, knnResult.docId)
            && Arrays.equals(vector, knnResult.vector)
            && (Float.compare(score, knnResult.score) == 0 || Math.abs(score - knnResult.score) <= delta);
    }
}
