/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.*;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import org.opensearch.threadpool.ThreadPool;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.io.IOException;

/**
 * Utility class for performing vector calculations and profiling operations
 * on collections of float arrays representing vectors. This class utilizes the singleton
 * pattern to maintain consistent state across operations within the OpenSearch cluster.
 *
 * VectorProfiler provides functionality to:
 * Calculate statistical measures (mean, variance, standard deviation) on vector collections
 * Sample vectors from larger collections for efficient processing
 * Track and record vectors during read and write operations
 * Save vector statistics to disk for later analysis
 */
@Log4j2
public class VectorProfiler<T extends Computation> {

    // Singleton instance
    private static VectorProfiler INSTANCE;

    // Default maximum number of vectors to sample per segment
    private static final int DEFAULT_MAX_SAMPLE_SIZE = 100;

    // Max number of vector elements to print in stats files
    private static final int MAX_VECTOR_ELEMENTS_TO_PRINT = 10;

    // Maps to store segment context information
    private static final ConcurrentHashMap<String, String> SEGMENT_BASE_NAMES = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, String> SEGMENT_SUFFIXES = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Path> SEGMENT_DIRECTORY_PATHS = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, float[]> SEGMENT_SUM_PER_DIMENSION = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Computation> SEGMENT_COMPUTATIONS = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Long> SEGMENT_COUNTS = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, List<float[]>> SEGMENT_SAMPLE_VECTORS = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Integer> SEGMENT_MAX_SAMPLE_SIZES = new ConcurrentHashMap<>();
    private ThreadPool threadPool;

    /**
     * Private constructor for singleton pattern
     */
    private VectorProfiler() {}

    public void initialize(ThreadPool threadPool) {
        this.threadPool = threadPool;
    }

    /**
     * Get the singleton instance of VectorProfiler
     *
     * @return Singleton instance of VectorProfiler
     */
    public static synchronized VectorProfiler getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new VectorProfiler();
        }
        return INSTANCE;
    }

    /**
     * Set the instance of VectorProfiler (primarily used for testing)
     *
     * @param instance VectorProfiler instance
     */
    public static synchronized void setInstance(VectorProfiler instance) {
        INSTANCE = instance;
    }

    /**
     * Calculates a statistical vector (mean, variance, etc.) based on a collection of input vectors
     * using the specified computation.
     *
     * @param <T> Type parameter extending Computation interface
     * @param vectors Collection of float arrays representing input vectors
     * @param computation The computation to be performed on the vectors
     * @return float array representing the calculated result vector
     * @throws IllegalArgumentException if vectors is null, empty, or contains vectors of different dimensions
     */
    public static <T extends Computation> float[] calculateVector(Collection<float[]> vectors, T computation) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors collection cannot be null or empty");
        }

        try {
            float[] firstVector = vectors.iterator().next();
            int dim = firstVector.length;

            float[] result = new float[dim];
            Arrays.fill(result, 0);

            for (float[] vec : vectors) {
                if (vec.length != dim) {
                    throw new IllegalArgumentException("All vectors must have same dimension");
                }
                for (int i = 0; i < dim; i++) {
                    result[i] = computation.apply(result[i], vec[i])[0];
                }
            }

            return computation.apply(result, vectors.size());
            // return result;
        } catch (Exception e) {
            log.error("Error in calculateVector: " + e.getMessage(), e);
            throw e;
        }
    }

    /**
     * Records vectors during read-time operations for a specific segment.
     * This allows tracking statistics of vectors that are used during query operations.
     *
     * @param <T> Type parameter extending Computation interface
     * @param segBaseName Base name of the segment
     * @param segSuffix Suffix of the segment
     * @param directoryPath Path to the directory where the segment is stored
     * @param vectors Collection of vectors to record
     * @param computation The computation to be performed on the vectors
     */
    public static <T extends Computation> void recordReadTimeVectors(
        String segBaseName,
        String segSuffix,
        Path directoryPath,
        Collection<float[]> vectors,
        T computation
    ) {

        if (vectors == null || vectors.isEmpty()) {
            log.debug("Vectors collection is null or empty");
            return;
        }

        try {
            int dim = vectors.iterator().next().length;
            log.debug("Recording read-time vectors for segment: {} with dimension: {}", segBaseName, dim);

            String contextKey = segBaseName + "_" + segSuffix;

            // Initialize segment context if it doesn't exist
            if (!SEGMENT_BASE_NAMES.containsKey(contextKey)) {
                SEGMENT_BASE_NAMES.put(contextKey, segBaseName);
                SEGMENT_SUFFIXES.put(contextKey, segSuffix);
                SEGMENT_DIRECTORY_PATHS.put(contextKey, directoryPath);
                SEGMENT_SUM_PER_DIMENSION.put(contextKey, new float[dim]);
                SEGMENT_COMPUTATIONS.put(contextKey, computation);
                SEGMENT_COUNTS.put(contextKey, 0L);
                SEGMENT_SAMPLE_VECTORS.put(contextKey, new ArrayList<>());
                SEGMENT_MAX_SAMPLE_SIZES.put(contextKey, DEFAULT_MAX_SAMPLE_SIZE);
            }

            addVectorsToSegment(contextKey, vectors);

            // Write vector statistics to file
            writeVectorStats(contextKey);
        } catch (Exception e) {
            log.error("Error in recordReadTimeVectors: " + e.getMessage(), e);
        }
    }

    /**
     * Add vectors to a segment's statistics and sample them as needed
     *
     * @param contextKey The key identifying the segment
     * @param vectors Collection of vectors to add
     */
    private static void addVectorsToSegment(String contextKey, Collection<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return;
        }

        List<float[]> sampleVectors = SEGMENT_SAMPLE_VECTORS.get(contextKey);
        int maxSampleSize = SEGMENT_MAX_SAMPLE_SIZES.get(contextKey);
        float[] sumPerDimension = SEGMENT_SUM_PER_DIMENSION.get(contextKey);
        Computation computation = SEGMENT_COMPUTATIONS.get(contextKey);
        long count = SEGMENT_COUNTS.get(contextKey);

        // Sample vectors if needed - using reservoir sampling to maintain a representative sample
        if (!vectors.isEmpty() && sampleVectors.size() < maxSampleSize) {
            Sampler sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
            int[] sampleIndices = sampler.sample(vectors.size(), Math.min(maxSampleSize - sampleVectors.size(), vectors.size()));

            List<float[]> vectorList = new ArrayList<>(vectors);
            for (int index : sampleIndices) {
                if (index < vectorList.size()) {
                    sampleVectors.add(vectorList.get(index).clone());
                }
            }
        }

        // Update running statistics for all vectors
        for (float[] vec : vectors) {
            for (int i = 0; i < sumPerDimension.length; i++) {
                sumPerDimension[i] = computation.apply(sumPerDimension[i], vec[i])[0];
            }
            count++;
        }

        // Update the maps with new values
        SEGMENT_SUM_PER_DIMENSION.put(contextKey, sumPerDimension);
        SEGMENT_SAMPLE_VECTORS.put(contextKey, sampleVectors);
        SEGMENT_COUNTS.put(contextKey, count);

        log.debug("Added {} vectors to segment {}. Total vectors: {}", vectors.size(), SEGMENT_BASE_NAMES.get(contextKey), count);
    }

    /**
     * Calculate the statistical result based on the computation type for a segment
     *
     * @param contextKey The key identifying the segment
     * @return Statistical result as float array
     */
    static float[] calculateResultForSegment(String contextKey) {
        long count = SEGMENT_COUNTS.getOrDefault(contextKey, 0L);
        if (count == 0) {
            log.debug("No vectors available (count is 0) for segment {}", SEGMENT_BASE_NAMES.get(contextKey));
            return null;
        }

        log.debug("Calculating vector statistics for {} vectors in segment {}", count, SEGMENT_BASE_NAMES.get(contextKey));

        float[] sumPerDimension = SEGMENT_SUM_PER_DIMENSION.get(contextKey);
        Computation computation = SEGMENT_COMPUTATIONS.get(contextKey);

        return computation.apply(sumPerDimension, count);
    }

    /**
     * Saves vector statistics from a segment write operation to a file.
     *
     * @param segmentWriteState Segment write state containing segment information
     * @param vectors Collection of vectors to save statistics for
     * @throws IOException If an error occurs during file writing
     */
    public static void saveVectorStats(SegmentWriteState segmentWriteState, Collection<float[]> vectors) throws IOException {
        if (vectors == null || vectors.isEmpty()) {
            log.debug("No vectors to save statistics for");
            return;
        }

        try {
            // Calculate all statistics
            float[] meanVector = calculateVector(vectors, StatisticalOperators.MEAN);
            float[] varianceVector = calculateVector(vectors, StatisticalOperators.VARIANCE);
            float[] stdDevVector = calculateVector(vectors, StatisticalOperators.STANDARD_DEVIATION);

            String statsFileName = IndexFileNames.segmentFileName(
                segmentWriteState.segmentInfo.name,
                segmentWriteState.segmentSuffix,
                "vectors_stats.txt"
            );

            Directory directory = segmentWriteState.directory;
            while (directory instanceof FilterDirectory) {
                directory = ((FilterDirectory) directory).getDelegate();
            }

            if (!(directory instanceof FSDirectory)) {
                throw new IOException("Expected FSDirectory but found " + directory.getClass().getSimpleName());
            }

            Path directoryPath = ((FSDirectory) directory).getDirectory();
            Path statsFile = directoryPath.resolve(statsFileName);

            // Create the parent directories if they don't exist
            Files.createDirectories(statsFile.getParent());

            // Build the statistics string
            StringBuilder sb = new StringBuilder();
            sb.append("=== Vector Statistics @ ").append(System.currentTimeMillis()).append(" ===\n");
            sb.append("Vector count: ").append(vectors.size()).append("\n");

            // Write mean vector
            sb.append("mean vector: [");
            appendVector(sb, meanVector);
            sb.append("]\n");

            // Write variance vector
            sb.append("variance vector: [");
            appendVector(sb, varianceVector);
            sb.append("]\n");

            // Write standard deviation vector
            sb.append("standard deviation vector: [");
            appendVector(sb, stdDevVector);
            sb.append("]\n\n");

            // Write to file
            Files.write(statsFile, sb.toString().getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE, StandardOpenOption.APPEND);

            log.debug("Saved vector statistics to {}", statsFile);
        } catch (Exception e) {
            log.error("Error saving vector statistics: " + e.getMessage(), e);
            throw e;
        }
    }

    /**
     * Writes vector statistics for a segment to a file
     *
     * @param contextKey The key identifying the segment
     */
    private static void writeVectorStats(String contextKey) {
        long count = SEGMENT_COUNTS.getOrDefault(contextKey, 0L);
        if (count == 0) {
            log.debug("No vectors to write statistics for in segment {}", SEGMENT_BASE_NAMES.get(contextKey));
            return;
        }

        try {
            String segBaseName = SEGMENT_BASE_NAMES.get(contextKey);
            String segSuffix = SEGMENT_SUFFIXES.get(contextKey);
            Path directoryPath = SEGMENT_DIRECTORY_PATHS.get(contextKey);

            String fileName = IndexFileNames.segmentFileName(segBaseName, segSuffix, "vectors_stats.txt");
            Path outputFile = directoryPath.resolve(fileName);

            // Sample vectors to compute statistics
            List<float[]> sampleVectors = SEGMENT_SAMPLE_VECTORS.get(contextKey);
            if (sampleVectors.isEmpty()) {
                log.debug("No sample vectors available for segment {}", segBaseName);
                return;
            }

            // Calculate statistics on sampled vectors
            float[] meanVector = calculateVector(sampleVectors, StatisticalOperators.MEAN);
            float[] varianceVector = calculateVector(sampleVectors, StatisticalOperators.VARIANCE);
            float[] stdDevVector = calculateVector(sampleVectors, StatisticalOperators.STANDARD_DEVIATION);

            // Create the parent directories if they don't exist
            Files.createDirectories(outputFile.getParent());

            // Build the statistics string
            StringBuilder sb = new StringBuilder();
            sb.append("=== Vector Statistics @ ").append(System.currentTimeMillis()).append(" ===\n");
            sb.append("Total vectors: ").append(count).append("\n");
            sb.append("Sampled vectors: ").append(sampleVectors.size()).append("\n");

            // Write mean vector
            sb.append("mean vector: [");
            appendVector(sb, meanVector);
            sb.append("]\n");

            // Write variance vector
            sb.append("variance vector: [");
            appendVector(sb, varianceVector);
            sb.append("]\n");

            // Write standard deviation vector
            sb.append("standard deviation vector: [");
            appendVector(sb, stdDevVector);
            sb.append("]\n\n");

            // Write to file
            Files.write(outputFile, sb.toString().getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE, StandardOpenOption.APPEND);

            log.debug("Wrote vector statistics to {}", outputFile);
        } catch (Exception e) {
            log.error("Error writing vector statistics for segment {}: {}", SEGMENT_BASE_NAMES.get(contextKey), e.getMessage(), e);
        }
    }

    /**
     * Helper method to append a vector to a StringBuilder.
     * For large vectors, only prints a subset of the vector elements.
     *
     * @param sb StringBuilder to append to
     * @param vector Vector to append
     */
    static void appendVector(StringBuilder sb, float[] vector) {
        if (vector == null || vector.length == 0) {
            sb.append("]");
            return;
        }

        // Only include the first few and last few elements if the vector is large
        if (vector.length <= MAX_VECTOR_ELEMENTS_TO_PRINT) {
            // Print all elements
            for (int i = 0; i < vector.length; i++) {
                sb.append(vector[i]);
                if (i < vector.length - 1) {
                    sb.append(", ");
                }
            }
        } else {
            // Print first few elements
            for (int i = 0; i < MAX_VECTOR_ELEMENTS_TO_PRINT / 2; i++) {
                sb.append(vector[i]).append(", ");
            }

            // Indicate truncation
            sb.append("... ");

            // Print last few elements
            for (int i = vector.length - MAX_VECTOR_ELEMENTS_TO_PRINT / 2; i < vector.length; i++) {
                sb.append(vector[i]);
                if (i < vector.length - 1) {
                    sb.append(", ");
                }
            }
        }
    }

    /**
     * Clear all segment contexts - mainly used for testing and cleanup
     */
    public static void clearSegmentContexts() {
        SEGMENT_BASE_NAMES.clear();
        SEGMENT_SUFFIXES.clear();
        SEGMENT_DIRECTORY_PATHS.clear();
        SEGMENT_SUM_PER_DIMENSION.clear();
        SEGMENT_COMPUTATIONS.clear();
        SEGMENT_COUNTS.clear();
        SEGMENT_SAMPLE_VECTORS.clear();
        SEGMENT_MAX_SAMPLE_SIZES.clear();
    }

    /**
     * Get sampled vectors for a specific segment
     *
     * @param contextKey The key identifying the segment
     * @return List of sampled vectors
     */
    public static List<float[]> getSampleVectorsForSegment(String contextKey) {
        List<float[]> samples = SEGMENT_SAMPLE_VECTORS.get(contextKey);
        if (samples == null) {
            return new ArrayList<>();
        }
        return new ArrayList<>(samples);
    }

    /**
     * Get the count of vectors in a specific segment
     *
     * @param contextKey The key identifying the segment
     * @return The count of vectors
     */
    public static long getSegmentVectorCount(String contextKey) {
        return SEGMENT_COUNTS.getOrDefault(contextKey, 0L);
    }

    /**
     * Get the segment base name for a specific segment
     *
     * @param contextKey The key identifying the segment
     * @return The segment base name
     */
    public static String getSegmentBaseName(String contextKey) {
        return SEGMENT_BASE_NAMES.get(contextKey);
    }

    /**
     * Get the segment suffix for a specific segment
     *
     * @param contextKey The key identifying the segment
     * @return The segment suffix
     */
    public static String getSegmentSuffix(String contextKey) {
        return SEGMENT_SUFFIXES.get(contextKey);
    }

    /**
     * Get the directory path for a specific segment
     *
     * @param contextKey The key identifying the segment
     * @return The directory path
     */
    public static Path getSegmentDirectoryPath(String contextKey) {
        return SEGMENT_DIRECTORY_PATHS.get(contextKey);
    }
}
