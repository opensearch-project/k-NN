/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.profiler;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.UUID;

/**
 * VectorProfiler is a singleton class that manages statistical aggregation of vector data across different fields.
 * It maintains dimension-specific statistics for each field, allowing for analysis of vector characteristics
 * across different segments of an index.
 *
 * The profiler collects and maintains statistical information about vectors in different dimensions,
 * which can be used for various analytical purposes such as understanding data distribution,
 * identifying patterns, and optimizing vector operations.
 */
@Getter
@Log4j2
public class VectorProfiler {
    private static VectorProfiler INSTANCE;
    private final Map<String, List<DimensionStatisticAggregator>> fieldToDimensionStats;

    private VectorProfiler() {
        this.fieldToDimensionStats = new HashMap<>();
    }

    public static synchronized VectorProfiler getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new VectorProfiler();
        }
        return INSTANCE;
    }

    /**
     * Profiles a collection of vectors for a specific field, computing statistical summaries
     * for each dimension. The statistics are aggregated across multiple segments of the index.
     *
     * @param fieldName The name of the field containing the vectors
     * @param vectors The collection of vectors to profile
     * @param dimensions The number of dimensions in the vectors
     */
    public void profileVectors(final String fieldName, final Collection<float[]> vectors, final int dimensions) {
        if (vectors == null || vectors.isEmpty()) {
            log.warn("No vectors to profile for field: {}", fieldName);
            return;
        }

        List<DimensionStatisticAggregator> aggregators = fieldToDimensionStats.computeIfAbsent(
            fieldName,
            k -> initializeDimensionAggregators(dimensions)
        );
        updateDimensionStatistics(aggregators, vectors);
    }

    private List<DimensionStatisticAggregator> initializeDimensionAggregators(final int dimensions) {
        List<DimensionStatisticAggregator> dimensionAggregators = new ArrayList<>(dimensions);
        for (int i = 0; i < dimensions; i++) {
            dimensionAggregators.add(new DimensionStatisticAggregator(i));
        }
        return dimensionAggregators;
    }

    // TODO: Attach segmentID to a specific ID rather than a random one
    // Currently using UUID for example usecase by utilizing random IDs
    private void updateDimensionStatistics(final List<DimensionStatisticAggregator> aggregators, final Collection<float[]> vectors) {
        String segmentId = UUID.randomUUID().toString();
        for (float[] vector : vectors) {
            for (int dim = 0; dim < Math.min(aggregators.size(), vector.length); dim++) {
                aggregators.get(dim).addValue(segmentId, vector[dim]);
            }
        }
    }

    /**
     * Retrieve statistical aggregators for a specific field.
     *
     * @param fieldName The name of the field
     * @return List of DimensionStatisticAggregator for the field, or null if field not found
     */
    public List<DimensionStatisticAggregator> getFieldStatistics(final String fieldName) {
        return fieldToDimensionStats.get(fieldName);
    }
}
