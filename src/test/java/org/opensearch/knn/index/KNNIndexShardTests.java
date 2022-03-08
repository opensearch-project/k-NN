/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.GRAPH_COUNT;

public class KNNIndexShardTests extends KNNSingleNodeTestCase {

    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    public void testGetIndexShard() throws InterruptedException, ExecutionException, IOException {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 2.5F, 3.5F });

        IndexShard indexShard = indexService.iterator().next();
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        assertEquals(indexShard, knnIndexShard.getIndexShard());
    }

    public void testGetIndexName() throws InterruptedException, ExecutionException, IOException {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 2.5F, 3.5F });

        IndexShard indexShard = indexService.iterator().next();
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        assertEquals(testIndexName, knnIndexShard.getIndexName());
    }

    public void testWarmup_emptyIndex() throws IOException {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);

        IndexShard indexShard = indexService.iterator().next();
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.warmup();
        assertNull(NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName));
    }

    public void testWarmup_shardPresentInCache() throws InterruptedException, ExecutionException, IOException {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 2.5F, 3.5F });

        searchKNNIndex(testIndexName, testFieldName, new float[] { 1.0f, 2.0f }, 1);
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName).get(GRAPH_COUNT));

        IndexShard indexShard = indexService.iterator().next();
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.warmup();
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName).get(GRAPH_COUNT));
    }

    public void testWarmup_shardNotPresentInCache() throws InterruptedException, ExecutionException, IOException {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        IndexShard indexShard;
        KNNIndexShard knnIndexShard;

        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 2.5F, 3.5F });
        client().admin().indices().prepareFlush(testIndexName).execute();

        indexShard = indexService.iterator().next();
        knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.warmup();
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName).get(GRAPH_COUNT));

        addKnnDoc(testIndexName, "2", testFieldName, new Float[] { 2.5F, 3.5F });
        indexShard = indexService.iterator().next();
        knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.warmup();
        assertEquals(2, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName).get(GRAPH_COUNT));
    }

    public void testGetHNSWPaths() throws IOException, ExecutionException, InterruptedException {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        IndexShard indexShard;
        KNNIndexShard knnIndexShard;
        Engine.Searcher searcher;
        Map<String, SpaceType> hnswPaths;

        indexShard = indexService.iterator().next();
        knnIndexShard = new KNNIndexShard(indexShard);

        searcher = indexShard.acquireSearcher("test-hnsw-paths-1");
        hnswPaths = knnIndexShard.getAllEnginePaths(searcher.getIndexReader());
        assertEquals(0, hnswPaths.size());
        searcher.close();

        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 2.5F, 3.5F });

        searcher = indexShard.acquireSearcher("test-hnsw-paths-2");
        hnswPaths = knnIndexShard.getAllEnginePaths(searcher.getIndexReader());
        assertEquals(1, hnswPaths.size());
        List<String> paths = new ArrayList<>(hnswPaths.keySet());
        assertTrue(paths.get(0).contains("hnsw") || paths.get(0).contains("hnswc"));
        searcher.close();
    }

    public void testGetEnginePaths() {
        // Check that the correct engine paths are being returned by the KNNIndexShard
        String segmentName = "_0";
        String fieldName = "test_field";
        String fileExt = ".test";
        SpaceType spaceType = SpaceType.L2;

        Set<String> includedFileNames = ImmutableSet.of(
            String.format("%s_111_%s%s", segmentName, fieldName, fileExt),
            String.format("%s_7_%s%s", segmentName, fieldName, fileExt),
            String.format("%s_53_%s%s", segmentName, fieldName, fileExt)
        );

        List<String> excludedFileNames = ImmutableList.of(
            String.format("_111_%s%s", fieldName, fileExt), // missing segment name
            String.format("%s_111_%s", segmentName, fileExt), // missing field name
            String.format("%s_111_%s.invalid", segmentName, fieldName) // missing extension
        );

        List<String> files = Stream.concat(includedFileNames.stream(), excludedFileNames.stream()).collect(Collectors.toList());

        KNNIndexShard knnIndexShard = new KNNIndexShard(null);

        Path path = Paths.get("");
        Map<String, SpaceType> included = knnIndexShard.getEnginePaths(files, segmentName, fieldName, fileExt, path, spaceType);

        assertEquals(includedFileNames.size(), included.size());
        included.keySet().forEach(o -> assertTrue(includedFileNames.contains(o)));
    }
}
