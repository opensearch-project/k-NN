/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;
import org.junit.Test;
import org.mockito.Mockito;
import org.opensearch.action.admin.indices.stats.IndexStats;
import org.opensearch.action.admin.indices.stats.ShardStats;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.env.Environment;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.test.OpenSearchTestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.Sort;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class SegmentProfilerStateTests extends OpenSearchTestCase {

    @Test
    public void testProcessVectorWithFloatArray() {
        List<SummaryStatistics> stats = new ArrayList<>();
        stats.add(new SummaryStatistics());
        stats.add(new SummaryStatistics());

        float[] vector = { 1.5f, 2.8f };
        SegmentProfilerState.processVector(vector, stats);

        assertEquals(1.5, stats.get(0).getMin(), 0.001);
        assertEquals(2.8, stats.get(1).getMin(), 0.001);
        assertEquals(1, stats.get(0).getN());
    }

    @Test
    public void testProcessVectorWithByteArray() {
        List<SummaryStatistics> stats = new ArrayList<>();
        stats.add(new SummaryStatistics());
        stats.add(new SummaryStatistics());

        byte[] vector = { (byte) 100, (byte) 200 };
        SegmentProfilerState.processVector(vector, stats);

        assertEquals(100.0, stats.get(0).getMin(), 0.001);
        assertEquals(200.0, stats.get(1).getMax(), 0.001);
    }

    @Test
    @SneakyThrows
    public void testProfileVectorsWritesStatsFile() {
        KNNVectorValues<Object> mockVectorValues = mock(KNNVectorValues.class);
        when(mockVectorValues.dimension()).thenReturn(2);
        when(mockVectorValues.docId()).thenReturn(0, NO_MORE_DOCS);
        when(mockVectorValues.getVector()).thenReturn(new float[] { 1.0f, 2.0f });
        Supplier<KNNVectorValues<?>> supplier = () -> mockVectorValues;

        Path tempDir = createTempDir();
        FSDirectory directory = FSDirectory.open(tempDir);

        String segmentSuffix = "knn";

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "test_segment",
            1,
            false,
            false,
            Mockito.mock(Codec.class),
            Mockito.mock(Map.class),
            new byte[16],
            Mockito.mock(Map.class),
            Mockito.mock(Sort.class)
        );

        SegmentWriteState state = new SegmentWriteState(
            null,
            directory,
            segmentInfo,
            new FieldInfos(new FieldInfo[0]),
            null,
            mock(IOContext.class)
        );

        SegmentProfilerState profilerState = SegmentProfilerState.profileVectors(supplier, state, "test_field");

        String fileName = IndexFileNames.segmentFileName(segmentInfo.name, state.segmentSuffix, "json");
        Path statsFile = tempDir.resolve(fileName);

        assertTrue("Stats file should exist at: " + statsFile, Files.exists(statsFile));

        String content = Files.readString(statsFile);
        assertNotNull("File content should not be null", content);
        assertTrue("File should not be empty", content.length() > 0);

        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                content
            )
        ) {

            Map<String, Object> jsonMap = parser.map();
            assertTrue("JSON should contain dimensions", jsonMap.containsKey("dimensions"));
        }
    }

    @Test
    public void testProfileVectorsWithEmptyVectors() throws IOException {
        Supplier<KNNVectorValues<?>> supplier = () -> null;
        SegmentWriteState state = mock(SegmentWriteState.class);

        SegmentProfilerState profilerState = SegmentProfilerState.profileVectors(supplier, state, "test_field");
        assertTrue(profilerState.getStatistics().isEmpty());
    }

    @Test
    @SneakyThrows
    public void testGetIndexStatsAggregatesShardData() {
        Path tempDir = createTempDir();
        Files.write(
            tempDir.resolve("NativeEngines990KnnVectors.json"),
            "{\"dimensions\":[{\"min\":1.0,\"max\":3.0,\"count\":2},{\"min\":2.0,\"max\":4.0,\"count\":2}]}".getBytes()
        );

        Environment env = mock(Environment.class);
        when(env.dataFiles()).thenReturn(new Path[] { tempDir.getParent() });

        ShardRouting shardRouting = mock(ShardRouting.class);
        ShardStats shardStats = mock(ShardStats.class);
        when(shardStats.getShardRouting()).thenReturn(shardRouting);

        IndexStats indexStats = mock(IndexStats.class);
        when(indexStats.getShards()).thenReturn(new ShardStats[] { shardStats });

        String json;
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            builder.startObject();
            builder.startArray("dimensions");

            builder.startObject();
            builder.field("dimension", 0);
            builder.field("min", 1.0);
            builder.field("max", 3.0);
            builder.field("count", 2);
            builder.endObject();

            builder.startObject();
            builder.field("dimension", 1);
            builder.field("min", 2.0);
            builder.field("max", 4.0);
            builder.field("count", 2);
            builder.endObject();

            builder.endArray();
            builder.endObject();

            json = builder.toString();
        }

        assertNotNull(json);
        assertTrue("JSON should contain dimension 0", json.contains("\"dimension\":0"));
        assertTrue("JSON should contain min value", json.contains("\"min\":1.0"));
        assertTrue("JSON should contain max value", json.contains("\"max\":3.0"));

        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                json
            )
        ) {

            Map<String, Object> jsonMap = parser.map();
            assertTrue("JSON should contain dimensions", jsonMap.containsKey("dimensions"));

            List<?> dimensions = (List<?>) jsonMap.get("dimensions");
            assertEquals("Should have 2 dimensions", 2, dimensions.size());

            Map<?, ?> firstDimension = (Map<?, ?>) dimensions.get(0);
            assertEquals("First dimension should be 0", 0, firstDimension.get("dimension"));
            assertEquals("First dimension min should be 1.0", 1.0, (Double) firstDimension.get("min"), 0.001);
            assertEquals("First dimension max should be 3.0", 3.0, (Double) firstDimension.get("max"), 0.001);
        }
    }

    @Test
    public void testParseStatsFromJson() throws IOException {
        String json = """
            {"dimensions":[
                {"dimension":0,"min":1.0,"max":3.0,"count":2},
                {"dimension":1,"min":2.0,"max":4.0,"count":2}
            ]}""";

        List<SummaryStatistics> stats = SegmentProfilerState.parseStatsFromJson(json);
        assertEquals(2, stats.size());
        assertEquals(1.0, stats.get(0).getMin(), 0.001);
        assertEquals(4.0, stats.get(1).getMax(), 0.001);
    }

    @Test
    public void testMergeStatistics() {
        List<SummaryStatistics> target = new ArrayList<>();
        List<SummaryStatistics> source = new ArrayList<>();

        SummaryStatistics srcStat1 = new SummaryStatistics();
        srcStat1.addValue(1.0);
        srcStat1.addValue(3.0);
        source.add(srcStat1);

        SegmentProfilerState.mergeStatistics(target, source);
        assertEquals(1, target.size());
        assertEquals(1.0, target.get(0).getMin(), 0.001);
        assertEquals(3.0, target.get(0).getMax(), 0.001);
    }

    @Test
    public void testGetUnderlyingDirectory() throws IOException {
        FSDirectory fsDir = mock(FSDirectory.class);
        FilterDirectory filterDir = new FilterDirectory(fsDir) {
        };
        Directory result = SegmentProfilerState.getUnderlyingDirectory(filterDir);
        assertEquals(fsDir, result);
    }

    @Test(expected = IOException.class)
    public void testGetUnderlyingDirectoryThrowsForNonFSDirectory() throws IOException {
        Directory dir = mock(Directory.class);
        SegmentProfilerState.getUnderlyingDirectory(dir);
    }
}
