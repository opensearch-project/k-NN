/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.sampler;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ReservoirSampler implements Sampler {
    private final Random random = new Random();

    @Override
    public int[] sample(int totalNumberOfVectors, int sampleSize) {
        if (totalNumberOfVectors <= sampleSize) {
            return IntStream.range(0, totalNumberOfVectors).toArray();
        }
        return reservoirSampleIndices(totalNumberOfVectors, sampleSize);
    }
    private int[] reservoirSampleIndices(int numVectors, int sampleSize) {
        int[] indices = IntStream.range(0, sampleSize).toArray();
        for (int i = sampleSize; i < numVectors; i++) {
            int j = random.nextInt(i + 1);
            if (j < sampleSize) {
                indices[j] = i;
            }
        }
        Arrays.sort(indices);
        return indices;
    }
}
