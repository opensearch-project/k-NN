/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.profiler;

import lombok.extern.log4j.Log4j2;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.UUID;

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

    public static synchronized void setInstance(VectorProfiler instance) {
        INSTANCE = instance;
    }

    public void setFieldToDimensionStats(Map<String, List<DimensionStatisticAggregator>> stats) {
        fieldToDimensionStats.clear();
        fieldToDimensionStats.putAll(stats);
    }

    public void processVectors(final String fieldName, final Collection<float[]> vectors) {
        validateVectors(vectors);

        float[] firstVector = vectors.iterator().next();
        int dimensions = firstVector.length;

        initializeDimensionAggregators(fieldName, dimensions);

        List<DimensionStatisticAggregator> aggregators = fieldToDimensionStats.get(fieldName);
        updateDimensionStatistics(aggregators, vectors);
    }

    private void initializeDimensionAggregators(final String fieldName, final int dimensions) {
        if (!fieldToDimensionStats.containsKey(fieldName)) {
            List<DimensionStatisticAggregator> dimensionAggregators = new ArrayList<>(dimensions);
            for (int i = 0; i < dimensions; i++) {
                dimensionAggregators.add(new DimensionStatisticAggregator(i));
            }
            fieldToDimensionStats.put(fieldName, dimensionAggregators);
        }
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

    public List<DimensionStatisticAggregator> getFieldStatistics(final String fieldName) {
        return fieldToDimensionStats.get(fieldName);
    }

    private void validateVectors(final Collection<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors collection cannot be null or empty");
        }
    }
}
