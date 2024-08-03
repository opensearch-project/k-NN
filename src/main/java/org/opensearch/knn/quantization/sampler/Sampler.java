/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

public interface Sampler {
    int[] sample(int totalNumberOfVectors, int sampleSize);
}
