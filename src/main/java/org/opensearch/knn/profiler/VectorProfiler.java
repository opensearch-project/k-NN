/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import org.apache.commons.math3.stat.descriptive.StatisticalSummary;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

@Log4j2
public class VectorProfiler {
    private static VectorProfiler INSTANCE;
    private static final int DEFAULT_SAMPLE_SIZE = 1000;
    private final Map<String, List<DimensionStatisticAggregator>> fieldToDimensionStats;
    private List<Computation> registeredComputations;

    public VectorProfiler() {
        this.registeredComputations = new ArrayList<>();
        this.fieldToDimensionStats = new HashMap<>();
        initialize();
    }

    public void initialize() {
        registerComputation(stats -> stats.getMean());
        registerComputation(stats -> stats.getVariance());
        registerComputation(stats -> Math.sqrt(stats.getVariance()));
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

    public void registerComputation(Computation computation) {
        registeredComputations.add(computation);
    }

    public void unregisterComputation(Computation computation) {
        registeredComputations.remove(computation);
    }

    public List<Computation> getRegisteredComputations() {
        return new ArrayList<>(registeredComputations);
    }

    public Map<Computation, float[]> sampleAndCompute(String fieldName, Collection<float[]> vectors, int... sampleSize) {
        validateVectors(vectors);

        Collection<float[]> sampledVectors = sampleVectors(vectors, sampleSize.length > 0 ? sampleSize[0] : DEFAULT_SAMPLE_SIZE);

        // Initialize dimension aggregators for the field if not exists
        if (!fieldToDimensionStats.containsKey(fieldName)) {
            int dimensions = sampledVectors.iterator().next().length;
            List<DimensionStatisticAggregator> dimensionAggregators = new ArrayList<>(dimensions);
            for (int i = 0; i < dimensions; i++) {
                dimensionAggregators.add(new DimensionStatisticAggregator(i));
            }
            fieldToDimensionStats.put(fieldName, dimensionAggregators);
        }

        // Update statistics for each dimension
        List<DimensionStatisticAggregator> aggregators = fieldToDimensionStats.get(fieldName);
        updateDimensionStatistics(aggregators, sampledVectors);

        // Compute results
        Map<Computation, float[]> results = new HashMap<>();
        for (Computation computation : registeredComputations) {
            try {
                results.put(computation, generateSampledDimensionVectors(aggregators, computation));
            } catch (IllegalArgumentException e) {
                log.error("Error performing computation for field {}: {}", fieldName, e.getMessage());
                throw e;
            }
        }

        return results;
    }

    private void updateDimensionStatistics(List<DimensionStatisticAggregator> aggregators, Collection<float[]> vectors) {
        int dimensions = aggregators.size();

        // Collect values for each dimension
        List<List<Float>> dimensionValues = new ArrayList<>(dimensions);
        for (int i = 0; i < dimensions; i++) {
            dimensionValues.add(new ArrayList<>());
        }

        // Group values by dimension
        for (float[] vector : vectors) {
            for (int dim = 0; dim < Math.min(dimensions, vector.length); dim++) {
                dimensionValues.get(dim).add(vector[dim]);
            }
        }

        // Update statistics for each dimension
        for (int dim = 0; dim < dimensions; dim++) {
            aggregators.get(dim).addSegmentStatistics(dimensionValues.get(dim));
        }
    }

    private float[] generateSampledDimensionVectors(List<DimensionStatisticAggregator> aggregators, Computation computation) {
        float[] results = new float[aggregators.size()];
        for (int dim = 0; dim < aggregators.size(); dim++) {
            StatisticalSummary stats = aggregators.get(dim).getAggregateStatistics();
            results[dim] = (float) computation.compute(stats);
        }
        return results;
    }

    private Collection<float[]> sampleVectors(Collection<float[]> vectors, int sampleSize) {
        Sampler sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
        int[] sampleIndices = sampler.sample(vectors.size(), Math.min(sampleSize, vectors.size()));

        List<float[]> vectorList = new ArrayList<>(vectors);
        List<float[]> sampledVectors = new ArrayList<>();

        for (int index : sampleIndices) {
            if (index < vectorList.size()) {
                sampledVectors.add(vectorList.get(index).clone());
            }
        }
        return sampledVectors;
    }

    private void validateVectors(Collection<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors collection cannot be null or empty");
        }
    }
}
