/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

/**
 * Interface defining computation operations for k-Nearest Neighbor (KNN) calculations.
 * This interface provides methods to perform numerical computations on floating point values
 * used in KNN algorithm implementations.
 */
public interface Computation {
    /**
     * Performs a computation operation on two float values.
     *
     * @param a The first float value
     * @param b The second float value
     * @return An array of float values representing the computation result
     */
    float[] apply(float a, float b);

    /**
     * Performs a computation operation using a sum array and a count.
     * Typically used for aggregating or averaging operations.
     *
     * @param sum Array containing sum values
     * @param count The number of elements or iterations
     * @return An array of float values representing the computation result
     */
    float[] apply(float[] sum, long count);
}