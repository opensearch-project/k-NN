/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FSDirectory;
import org.opensearch.action.admin.indices.stats.IndexStats;
import org.opensearch.action.admin.indices.stats.ShardStats;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.env.Environment;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * This class handles the profiling and statistical analysis of KNN vector segments
 * in OpenSearch. It provides functionality to collect, process, and store statistical
 * information about vector dimensions across different shards and segments.
 */
@Log4j2
public class SegmentProfilerState {
    private static final String VECTOR_STATS_EXTENSION = "json";
    private static final String VECTOR_OUTPUT_FILE = "NativeEngines990KnnVectors";
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("#.####");
    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ISO_INSTANT;

    @Getter
    private final List<SummaryStatistics> statistics;

    /**
     * Constructor initializing the statistics collection
     * @param statistics List of summary statistics for vector dimensions
     */
    public SegmentProfilerState(final List<SummaryStatistics> statistics) {
        this.statistics = statistics;
    }

    /**
     * Writes statistical data to a JSON file.
     * Stores raw statistics data to disk for later retrieval.
     * Called during vector indexing/processing.
     * @param outputFile Path to output file
     * @param statistics List of statistics to write
     * @param fieldName Name of the field being processed
     * @param vectorCount Total number of vectors
     */
    private static void writeStatsToFile(Path outputFile, List<SummaryStatistics> statistics, String fieldName, int vectorCount)
        throws IOException {
        // Create parent directories if they don't exist
        Files.createDirectories(outputFile.getParent());

        try (XContentBuilder jsonBuilder = XContentFactory.jsonBuilder()) {
            // Build JSON structure
            jsonBuilder.prettyPrint()
                .startObject()
                // Add metadata
                .field("timestamp", ISO_FORMATTER.format(Instant.now()))
                .field("fieldName", fieldName)
                .field("vectorCount", vectorCount)
                .field("dimension", statistics.size())
                .startArray("dimensions");
            // Add statistics for each dimension
            for (int i = 0; i < statistics.size(); i++) {
                SummaryStatistics stats = statistics.get(i);
                jsonBuilder.startObject()
                    .field("dimension", i)
                    .field("count", stats.getN())
                    .field("min", formatDouble(stats.getMin()))
                    .field("max", formatDouble(stats.getMax()))
                    .field("sum", formatDouble(stats.getSum()))
                    .field("mean", formatDouble(stats.getMean()))
                    .field("standardDeviation", formatDouble(Math.sqrt(stats.getVariance())))
                    .field("variance", formatDouble(stats.getVariance()))
                    .endObject();
            }

            jsonBuilder.endArray().endObject();
            Files.write(
                outputFile,
                jsonBuilder.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
            );
        }
    }

    /**
     * Profiles vectors in a segment and collects statistical information
     * @param knnVectorValuesSupplier Supplier for vector values
     * @param segmentWriteState State of the segment being written
     * @param fieldName Name of the field being processed
     * @return SegmentProfilerState containing collected statistics
     */
    public static SegmentProfilerState profileVectors(
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        final SegmentWriteState segmentWriteState,
        final String fieldName
    ) throws IOException {
        // Get vector values from the supplier
        KNNVectorValues<?> vectorValues = knnVectorValuesSupplier.get();
        if (vectorValues == null) {
            return new SegmentProfilerState(new ArrayList<>());
        }
        // Initialize new profiler state and vector values
        SegmentProfilerState profilerState = new SegmentProfilerState(new ArrayList<>());
        KNNCodecUtil.initializeVectorValues(vectorValues);
        int dimension = vectorValues.dimension();
        int vectorCount = 0;

        // Create statistics objects for each dimension
        for (int i = 0; i < dimension; i++) {
            profilerState.statistics.add(new SummaryStatistics());
        }

        // Process each vector in the segment
        while (vectorValues.docId() != NO_MORE_DOCS) {
            vectorCount++;
            Object vector = vectorValues.getVector();
            processVector(vector, profilerState.statistics);
            vectorValues.nextDoc();
        }

        // Generate filename and write statistics to file
        String statsFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            VECTOR_STATS_EXTENSION
        );

        Directory directory = getUnderlyingDirectory(segmentWriteState.directory);
        Path statsFile = ((FSDirectory) directory).getDirectory().resolve(statsFileName);
        writeStatsToFile(statsFile, profilerState.statistics, fieldName, vectorCount);

        return profilerState;
    }

    /**
     * Generates index-level statistics and writes them to the XContentBuilder.
     * Aggregates data from all shards and provides current view
     * @param indexStats Statistics for the index
     * @param builder XContentBuilder for response construction
     * @param environment OpenSearch environment
     */
    public static void getIndexStats(IndexStats indexStats, XContentBuilder builder, Environment environment) throws IOException {
        try {
            // Build index summary section
            builder.startObject("index_summary")
                .field("doc_count", indexStats.getTotal().getDocs().getCount())
                .field("size_in_bytes", indexStats.getTotal().getStore().getSizeInBytes())
                .field("timestamp", ISO_FORMATTER.format(Instant.now()))
                .endObject();

            builder.startObject("vector_stats").field("sample_size", indexStats.getTotal().getDocs().getCount());

            builder.startObject("summary_stats");
            List<SummaryStatistics> stats = getSummaryStatisticsForIndex(indexStats, environment);

            if (!stats.isEmpty()) {
                // Write dimension-wise statistics
                builder.startArray("dimensions");
                for (int i = 0; i < stats.size(); i++) {
                    SummaryStatistics dimStats = stats.get(i);
                    builder.startObject()
                        .field("dimension", i)
                        .field("count", dimStats.getN())
                        .field("min", formatDouble(dimStats.getMin()))
                        .field("max", formatDouble(dimStats.getMax()))
                        .field("sum", formatDouble(dimStats.getSum()))
                        .field("mean", formatDouble(dimStats.getMean()))
                        .field("standardDeviation", formatDouble(dimStats.getStandardDeviation()))
                        .field("variance", formatDouble(dimStats.getVariance()))
                        .endObject();
                }
                builder.endArray();
            } else {
                builder.field("status", "No statistics available");
            }

            builder.endObject().endObject();

        } catch (Exception e) {
            builder.startObject("error").field("message", "Failed to get statistics: " + e.getMessage()).endObject();
        }
    }

    /**
     * Collects summary statistics for an entire index
     * @param indexStats Statistics for the index
     * @param environment OpenSearch environment
     * @return List of summary statistics for each dimension
     */
    private static List<SummaryStatistics> getSummaryStatisticsForIndex(IndexStats indexStats, Environment environment) {
        List<SummaryStatistics> stats = new ArrayList<>();
        ShardStats[] shardStats = indexStats.getShards();

        if (shardStats != null) {
            // Process each shard in the index
            for (ShardStats shard : shardStats) {
                try {
                    Path indexPath = getShardIndexPath(shard, environment);
                    if (Files.exists(indexPath)) {
                        processShardDirectory(indexPath, stats);
                    }
                } catch (Exception e) {
                    log.error("Error processing shard stats", e);
                }
            }
        }
        return stats;
    }

    /**
     * Determines the path to a shard's index directory
     * @param shard Shard statistics
     * @param environment OpenSearch environment
     * @return Path to the shard's index directory
     */
    private static Path getShardIndexPath(ShardStats shard, Environment environment) {
        int shardId = shard.getShardRouting().shardId().getId();
        String indexUUID = shard.getShardRouting().shardId().getIndex().getUUID();
        return environment.dataFiles()[0].resolve("nodes")
            .resolve("0")
            .resolve("indices")
            .resolve(indexUUID)
            .resolve(String.valueOf(shardId))
            .resolve("index");
    }

    /**
     * Processes statistics files in a shard directory
     * @param indexPath Path to the index directory
     * @param stats List to store collected statistics
     */
    private static void processShardDirectory(Path indexPath, List<SummaryStatistics> stats) throws IOException {
        Files.list(indexPath).filter(path -> path.getFileName().toString().contains(VECTOR_OUTPUT_FILE)).forEach(path -> {
            try {
                String jsonContent = Files.readString(path);
                List<SummaryStatistics> shardStats = parseStatsFromJson(jsonContent);
                mergeStatistics(stats, shardStats);
            } catch (IOException e) {
                log.error("Error processing file: " + path, e);
            }
        });
    }

    /**
     * Merges statistics from source into target
     * @param target Target statistics list
     * @param source Source statistics list
     */
    static void mergeStatistics(List<SummaryStatistics> target, List<SummaryStatistics> source) {
        if (target.isEmpty()) {
            for (SummaryStatistics sourceStat : source) {
                SummaryStatistics newStat = new SummaryStatistics();
                newStat.addValue(sourceStat.getMin());
                if (sourceStat.getN() > 1) {
                    newStat.addValue(sourceStat.getMax());
                }
                target.add(newStat);
            }
        } else {
            for (int i = 0; i < target.size(); i++) {
                SummaryStatistics targetStat = target.get(i);
                SummaryStatistics sourceStat = source.get(i);

                // Add all values from source statistics
                if (sourceStat.getN() > 0) {
                    targetStat.addValue(sourceStat.getMin());
                    if (sourceStat.getN() > 1) {
                        targetStat.addValue(sourceStat.getMax());
                    }
                }
            }
        }
    }

    /**
     * Parses statistics from JSON content
     * @param jsonContent JSON string containing statistics
     * @return List of parsed summary statistics
     */
    static List<SummaryStatistics> parseStatsFromJson(String jsonContent) throws IOException {
        List<SummaryStatistics> statistics = new ArrayList<>();

        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                jsonContent
            )
        ) {

            parseJsonContent(parser, statistics);
        }
        return statistics;
    }

    /**
     * Parses JSON content and updates statistics
     * @param parser XContentParser for JSON content
     * @param statistics List of statistics to update
     */
    private static void parseJsonContent(XContentParser parser, List<SummaryStatistics> statistics) throws IOException {
        XContentParser.Token token;
        String currentFieldName = null;

        while ((token = parser.nextToken()) != null) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if ("dimensions".equals(currentFieldName) && token == XContentParser.Token.START_ARRAY) {
                parseDimensions(parser, statistics);
            }
        }
    }

    /**
     * Parses dimensions from JSON content
     * @param parser XContentParser for JSON content
     * @param statistics List of statistics to update
     */
    private static void parseDimensions(XContentParser parser, List<SummaryStatistics> statistics) throws IOException {
        while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
            if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                statistics.add(parseDimensionStats(parser));
            }
        }
    }

    /**
     * Parses a dimension's statistics from JSON content
     * @param parser XContentParser for JSON content
     * @return SummaryStatistics for the parsed dimension
     */
    private static SummaryStatistics parseDimensionStats(XContentParser parser) throws IOException {
        SummaryStatistics stats = new SummaryStatistics();
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0;
        long count = 0;

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case "min":
                    min = parser.doubleValue();
                    break;
                case "max":
                    max = parser.doubleValue();
                    break;
                case "sum":
                    sum = parser.doubleValue();
                    break;
                case "count":
                    count = parser.longValue();
                    break;
            }
        }

        if (count > 0) {
            stats.addValue(min);
            if (count > 1) {
                stats.addValue(max);
            }
            if (count > 2) {
                double remainingMean = (sum - min - max) / (count - 2);
                for (int i = 0; i < count - 2; i++) {
                    stats.addValue(remainingMean);
                }
            }
        }

        return stats;
    }

    /**
     * Processes a vector and updates statistics
     * @param vector Vector to process (float[] or byte[])
     * @param statistics List of statistics to update
     */
    static <T> void processVector(T vector, List<SummaryStatistics> statistics) {
        if (vector instanceof float[]) {
            float[] floatVector = (float[]) vector;
            for (int j = 0; j < floatVector.length; j++) {
                statistics.get(j).addValue(floatVector[j]);
            }
        } else if (vector instanceof byte[]) {
            byte[] byteVector = (byte[]) vector;
            for (int j = 0; j < byteVector.length; j++) {
                statistics.get(j).addValue(byteVector[j] & 0xFF);
            }
        }
    }

    /**
     * Gets the underlying FSDirectory from a potentially wrapped Directory
     * @param directory Input directory
     * @return Underlying FSDirectory
     */
    static Directory getUnderlyingDirectory(Directory directory) throws IOException {
        while (directory instanceof FilterDirectory) {
            directory = ((FilterDirectory) directory).getDelegate();
        }
        if (!(directory instanceof FSDirectory)) {
            throw new IOException("Expected FSDirectory but found " + directory.getClass().getSimpleName());
        }
        return directory;
    }

    /**
     * Formats a double value according to the specified decimal format
     * @param value Double value to format
     * @return Formatted double value
     */
    public static double formatDouble(double value) {
        return Math.round(value * 10000.0) / 10000.0;
    }
}
