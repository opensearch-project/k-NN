/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

@Log4j2
public class VectorProfiler {

    private static VectorProfiler INSTANCE;
    private static final int DEFAULT_SAMPLE_SIZE = 100;
    private List<Computation> registeredComputations;

    public VectorProfiler() {
        this.registeredComputations = new ArrayList<>();
    }

    public void initialize() {
        // Initialize default computations
        registeredComputations.add(StatisticalOperators.MEAN);
        registeredComputations.add(StatisticalOperators.VARIANCE);
        registeredComputations.add(StatisticalOperators.STANDARD_DEVIATION);
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

    public Map<Computation, float[]> sampleAndCompute(
            Collection<float[]> vectors,
            int... sampleSize) {

        validateVectors(vectors);

        Collection<float[]> sampledVectors = sampleVectors(
                vectors,
                sampleSize.length > 0 ? sampleSize[0] : DEFAULT_SAMPLE_SIZE
        );

        Map<Computation, float[]> results = new HashMap<>();
        for (Computation computation : registeredComputations) {
            try {
                results.put(computation, generateSampledDimensionVectors(sampledVectors, computation));
            } catch (IllegalArgumentException e) {
                log.error("Error performing computation: " + e.getMessage());
                throw e;
            }
        }
        return results;
    }

    private float[] generateSampledDimensionVectors(
            Collection<float[]> vectors,
            Computation computation) {

        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors collection cannot be null or empty");
        }

        float[] firstVector = vectors.iterator().next();
        int dim = firstVector.length;
        int numVectors = vectors.size();

        // Create rotated matrix where each row represents a dimension
        float[][] rotatedMatrix = new float[dim][numVectors];

        // Fill the rotated matrix
        int vectorIndex = 0;
        for (float[] vec : vectors) {
            for (int dimIndex = 0; dimIndex < Math.min(dim, vec.length); dimIndex++) {
                rotatedMatrix[dimIndex][vectorIndex] = vec[dimIndex];
            }
            vectorIndex++;
        }

        // Process each dimension
        float[] result = new float[dim];
        for (int dimIndex = 0; dimIndex < dim; dimIndex++) {
            // Pass the entire dimension vector to compute
            result[dimIndex] = computation.compute(rotatedMatrix[dimIndex])[0];
        }

        return result;
    }

    private Collection<float[]> sampleVectors(
            Collection<float[]> vectors,
            int sampleSize) {

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