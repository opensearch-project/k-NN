/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * SegmentProfilerState is responsible for analyzing and profiling vector data within segments.
 * This class calculates statistical measurements for each dimension of the vectors in a segment.
 */
@Log4j2
@AllArgsConstructor
public class SegmentProfilerState implements Serializable {

    // Stores statistical summaries for each dimension of the vectors
    @Getter
    private final List<SummaryStatistics> statistics;

    @Getter
    private final int dimension;

    /**
     * Profiles vectors in a segment by analyzing their statistical values
     *
     * @param knnVectorValuesSupplier
     * @return SegmentProfilerState
     * @throws IOException
     */
    public static SegmentProfilerState profileVectors(final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier) throws IOException {
        KNNVectorValues<?> vectorValues = knnVectorValuesSupplier.get();

        if (vectorValues == null) {
            log.info("No vector values available");
            return new SegmentProfilerState(new ArrayList<>(), 0);
        }

        // Initialize vector values
        KNNCodecUtil.initializeVectorValues(vectorValues);
        List<SummaryStatistics> statistics = new ArrayList<>();

        // Return empty state if no documents are present
        if (vectorValues.docId() == NO_MORE_DOCS) {
            log.info("No vectors to profile");
            return new SegmentProfilerState(statistics, vectorValues.dimension());
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
            processVectors(vectorValues.getVector(), statistics);
        }

        log.info("Vector profiling completed - processed {} vectors", vectorCount);

        logDimensionStatistics(statistics, dimension);

        return new SegmentProfilerState(statistics, vectorValues.dimension());
    }

    /**
     * Helper method to process a vector and update statistics
     *
     * @param vector
     * @param statistics
     */
    private static <T> void processVectors(T vector, List<SummaryStatistics> statistics) {
        if (vector instanceof float[]) {
            processFloatVector((float[]) vector, statistics);
        } else if (vector instanceof byte[]) {
            processByteVector((byte[]) vector, statistics);
        } else {
            log.warn("Unsupported vector type: {}.", vector.getClass());
        }
    }

    /**
     * Processes a float vector by updating the statistical summaries for each dimension
     *
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
     *
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
     *
     * @param statistics
     * @param dimension
     */
    private static void logDimensionStatistics(final List<SummaryStatistics> statistics, final int dimension) {
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

    /**
     * Converts the SegmentProfilerState to a byte array for serialization
     * @return
     */
    public byte[] toByteArray() {
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            out.writeVInt(dimension);
            out.writeVInt(statistics.size());
            for (SummaryStatistics stat : statistics) {
                out.writeDouble(stat.getMean());
                out.writeDouble(stat.getVariance());
                out.writeVLong(stat.getN());
                out.writeDouble(stat.getMin());
                out.writeDouble(stat.getMax());
                out.writeDouble(stat.getSum());
            }
            return out.bytes().toBytesRef().bytes;
        } catch (IOException e) {
            throw new RuntimeException("Failed to serialize SegmentProfilerState", e);
        }
    }

    /**
     * Deserializes a SegmentProfilerState from a byte array
     * @param bytes
     * @return
     */
    public static SegmentProfilerState fromBytes(byte[] bytes) {
        try (StreamInput input = StreamInput.wrap(bytes)) {
            int dimension = input.readVInt();
            int statsSize = input.readVInt();
            List<SummaryStatistics> statistics = new ArrayList<>(statsSize);

            for (int i = 0; i < statsSize; i++) {
                SummaryStatistics stat = new SummaryStatistics();
                stat.addValue(input.readDouble());
                stat.addValue(input.readDouble());
                long n = input.readVLong();
                stat.addValue(input.readDouble());
                stat.addValue(input.readDouble());
                stat.addValue(input.readDouble());
                statistics.add(stat);
            }

            return new SegmentProfilerState(statistics, dimension);
        } catch (IOException e) {
            throw new RuntimeException("Failed to deserialize SegmentProfilerState", e);
        }
    }
}
