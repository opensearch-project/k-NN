/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * SegmentProfilerState is responsible for analyzing and profiling vector data within segments.
 * This class calculates statistical measurements for each dimension of the vectors in a segment.
 */
@Log4j2
public class SegmentProfilerState {

    // Stores statistical summaries for each dimension of the vectors
    @Getter
    private final List<SummaryStatistics> statistics;

    /**
     * Constructor to initialize the SegmentProfilerState
     * @param statistics
     */
    public SegmentProfilerState(final List<SummaryStatistics> statistics) {
        this.statistics = statistics;
    }

    /**
     * Profiles vectors in a segment by analyzing their statistical values
     * @param knnVectorValuesSupplier
     * @return SegmentProfilerState
     * @throws IOException
     */
    public static SegmentProfilerState profileVectors(final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier) throws IOException {
        KNNVectorValues<?> vectorValues = knnVectorValuesSupplier.get();

        if (vectorValues == null) {
            log.info("No vector values available");
            return new SegmentProfilerState(new ArrayList<>());
        }

        // Initialize vector values
        KNNCodecUtil.initializeVectorValues(vectorValues);
        List<SummaryStatistics> statistics = new ArrayList<>();

        // Return empty state if no documents are present
        if (vectorValues.docId() == NO_MORE_DOCS) {
            log.info("No vectors to profile");
            return new SegmentProfilerState(statistics);
        }

        int dimension = vectorValues.dimension();
        log.info("Starting vector profiling with dimension: {}", dimension);

        // Initialize statistics collectors for each dimension
        for (int i = 0; i < dimension; i++) {
            statistics.add(new SummaryStatistics());
        }

        // Process all vectors
        int vectorCount = 0;
        for (int doc = vectorValues.docId(); doc != NO_MORE_DOCS; doc = vectorValues.nextDoc()) {
            vectorCount++;
            Object vector = vectorValues.getVector();
            processVectors(vector, statistics);
        }

        log.info("Vector profiling completed - processed {} vectors", vectorCount);
        logDimensionStatistics(statistics, dimension);

        return new SegmentProfilerState(statistics);
    }

    /**
     * Helper method to process a vector and update statistics
     * @param vector
     * @param statistics
     */
    private static void processVectors(Object vector, List<SummaryStatistics> statistics) {
        if (vector instanceof float[]) {
            processFloatVector((float[]) vector, statistics);
        } else if (vector instanceof byte[]) {
            processByteVector((byte[]) vector, statistics);
        } else {
            throw new IllegalArgumentException("Unsupported vector type.");
        }
    }

    /**
     * Processes a float vector by updating the statistical summaries for each dimension
     * @param vector
     * @param statistics
     */
    private static void processFloatVector(float[] vector, List<SummaryStatistics> statistics) {
        for (int j = 0; j < vector.length; j++) {
            statistics.get(j).addValue(vector[j]);
        }
    }

    /**
     * Processes a byte vector by updating the statistical summaries for each dimension
     * @param vector
     * @param statistics
     */
    private static void processByteVector(byte[] vector, List<SummaryStatistics> statistics) {
        for (int j = 0; j < vector.length; j++) {
            statistics.get(j).addValue(vector[j] & 0xFF);
        }
    }

    /**
     * Helper method to log statistics for each dimension
     * @param statistics
     * @param dimension
     */
    private static void logDimensionStatistics(List<SummaryStatistics> statistics, int dimension) {
        for (int i = 0; i < dimension; i++) {
            SummaryStatistics stats = statistics.get(i);
            log.info(
                "Dimension {} stats: mean={}, std={}, min={}, max={}",
                i,
                stats.getMean(),
                stats.getStandardDeviation(),
                stats.getMin(),
                stats.getMax()
            );
        }
    }
}
