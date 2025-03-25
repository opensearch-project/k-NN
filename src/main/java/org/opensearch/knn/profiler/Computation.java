/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

/**
 * Interface defining computation operations for k-Nearest Neighbor (KNN) calculations.
 * This interface provides methods to perform numerical computations on floating point values
 * used in KNN algorithm implementations.
 */
public interface Computation {
    /**
     * Performs a computation operation on a vector of float values.
     *
     * @param perDimensionVector Array containing values for each dimension
     * @return An array of float values representing the computation result
     */
    float[] compute(float[] perDimensionVector);
}

