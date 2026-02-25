/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

public interface VectorProducer {
    int getVectorCount();

    float[] getVector(int id);
}
