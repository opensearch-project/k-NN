/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;

/**
 * Interface defining computation operations for k-Nearest Neighbor (KNN) calculations.
 * This interface provides methods to perform numerical computations on floating point values
 * used in KNN algorithm implementations.
 */

public interface Computation {
    /**
     * Performs a computation operation on statistical summary data.
     *
     * @param stats StatisticalSummary containing the aggregated data
     * @return The result of the computation
     */
    double compute(StatisticalSummary stats);
}
