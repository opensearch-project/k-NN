/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mockito;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.index.IndexService;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.mockito.Mockito.when;
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
        updateIndexSetting(testIndexName, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0).build());
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
        updateIndexSetting(testIndexName, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0).build());
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

    public void testGetAllEngineFileContexts() throws IOException, ExecutionException, InterruptedException {
        IndexService indexService = createKNNIndex(testIndexName);
        assertThrows(
            IllegalArgumentException.class,
            () -> createKnnIndexMapping(testIndexName, testFieldName, dimensions, KNNEngine.NMSLIB)
        );
    }

    @SneakyThrows
    public void testGetEngineFileContexts() {
        // Check that the correct engine paths are being returned by the KNNIndexShard
        String segmentName = "_0";
        String fieldName = "test_field";
        String fileExt = ".faiss";
        SpaceType spaceType = SpaceType.L2;
        String modelId = "test-model";
        VectorDataType vectorDataType = VectorDataType.FLOAT;

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
        String[] filesArray = files.toArray(new String[files.size()]);
        KNNIndexShard knnIndexShard = new KNNIndexShard(null);

        final Directory dummyDirectory = Mockito.mock(Directory.class);
        final Codec codec = Mockito.mock(Codec.class);

        final SegmentInfo segmentInfo = new SegmentInfo(
            dummyDirectory,
            Version.LATEST,
            null,
            segmentName,
            0,
            true,
            false,
            codec,
            Collections.emptyMap(),
            new byte[StringHelper.ID_LENGTH],
            Collections.emptyMap(),
            null
        );

        final CompoundDirectory directory = Mockito.mock(KNN80CompoundDirectory.class);
        final CompoundFormat compoundFormat = Mockito.mock(CompoundFormat.class);
        when(codec.compoundFormat()).thenReturn(compoundFormat);
        when(compoundFormat.getCompoundReader(dummyDirectory, segmentInfo)).thenReturn(directory);
        when(directory.listAll()).thenReturn(filesArray);

        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, -1, 0, 0, null);
        List<KNNIndexShard.EngineFileContext> included = knnIndexShard.getEngineFileContexts(
            segmentCommitInfo,
            fieldName,
            KNNEngine.FAISS,
            spaceType,
            modelId,
            vectorDataType
        );

        assertEquals(includedFileNames.size(), included.size());
        included.stream().map(KNNIndexShard.EngineFileContext::getVectorFileName).forEach(o -> assertTrue(includedFileNames.contains(o)));
    }

    @SneakyThrows
    public void testClearCache_emptyIndex() {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);

        IndexShard indexShard = indexService.iterator().next();
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.clearCache();
        assertNull(NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName));
    }

    @SneakyThrows
    public void testClearCache_shardPresentInCache() {
        IndexService indexService = createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        updateIndexSetting(testIndexName, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0).build());
        addKnnDoc(testIndexName, String.valueOf(randomInt()), testFieldName, new Float[] { randomFloat(), randomFloat() });

        IndexShard indexShard = indexService.iterator().next();
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.warmup();
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName).get(GRAPH_COUNT));

        knnIndexShard.clearCache();
        assertNull(NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName));
    }
}
