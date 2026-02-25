/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import java.util.Arrays;

public class IDVectorProducer implements VectorProducer {
    int dimension;
    int vectorCount;

    public IDVectorProducer(int dimension, int vectorCount) {
        this.dimension = dimension;
        this.vectorCount = vectorCount;
    }

    @Override
    public int getVectorCount() {
        return vectorCount;
    }

    @Override
    public float[] getVector(int id) {
        float[] indexVector = new float[dimension];
        Arrays.fill(indexVector, (float) id);
        return indexVector;
    }
}
