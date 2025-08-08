/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.memory;

import com.google.common.cache.CacheStats;
import com.google.common.util.concurrent.UncheckedExecutionException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IndexInput;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import lombok.SneakyThrows;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.exception.OutOfNativeMemoryException;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchSingleNodeTestCase;
import org.opensearch.threadpool.Scheduler.Cancellable;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Set;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.spy;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.GRAPH_COUNT;
import static org.opensearch.knn.plugin.stats.StatNames.GRAPH_MEMORY_USAGE;

public class NativeMemoryCacheManagerTests extends OpenSearchSingleNodeTestCase {

    private ThreadPool threadPool;

    @Mock
    protected ClusterService clusterService;
    @Mock
    protected ClusterSettings clusterSettings;

    protected AutoCloseable openMocks;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        defaultClusterSettings.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(s -> s.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
        threadPool = new ThreadPool(Settings.builder().put("node.name", "NativeMemoryCacheManagerTests").build());
        NativeMemoryCacheManager.setThreadPool(threadPool);
    }

    @After
    public void shutdown() throws Exception {
        terminate(threadPool);
        tearDown();
    }

    @Override
    public void tearDown() throws Exception {
        // Clear out persistent metadata
        ClusterUpdateSettingsRequest clusterUpdateSettingsRequest = new ClusterUpdateSettingsRequest();
        Settings circuitBreakerSettings = Settings.builder().putNull(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED).build();
        clusterUpdateSettingsRequest.persistentSettings(circuitBreakerSettings);
        client().admin().cluster().updateSettings(clusterUpdateSettingsRequest).get();
        NativeMemoryCacheManager.getInstance().close();
        super.tearDown();
    }

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return Collections.singletonList(KNNPlugin.class);
    }

    public void testRebuildCache() throws ExecutionException, InterruptedException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        Cancellable task1 = nativeMemoryCacheManager.getMaintenanceTask();
        assertNotNull(task1);

        // Put entry in cache and check that the weight matches
        int size = 10;
        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test", size);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        assertEquals(size, nativeMemoryCacheManager.getCacheSizeInKilobytes());

        // Call rebuild and check total weight is at 0
        nativeMemoryCacheManager.rebuildCache();

        // Sleep for a second or two so that the executor can invalidate all entries
        Thread.sleep(2000);

        assertTrue(task1.isCancelled());
        assertNotNull(nativeMemoryCacheManager.getMaintenanceTask());

        assertEquals(0, nativeMemoryCacheManager.getCacheSizeInKilobytes());
        nativeMemoryCacheManager.close();
    }

    public void testGetCacheSizeInKilobytes() throws ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        assertEquals(0, nativeMemoryCacheManager.getCacheSizeInKilobytes());

        // Put 2 entries in cache and check that the weight matches
        int size1 = 10;
        int size2 = 20;
        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent("test-1", size1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent("test-2", size2);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, true);

        assertEquals(size1 + size2, nativeMemoryCacheManager.getCacheSizeInKilobytes());
        nativeMemoryCacheManager.close();
    }

    public void testGetCacheSizeAsPercentage() throws ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        long maxWeight = nativeMemoryCacheManager.getMaxCacheSizeInKilobytes();
        int entryWeight = (int) (maxWeight / 3);

        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test-1", entryWeight, 0);

        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        assertEquals(100 * (float) entryWeight / (float) maxWeight, nativeMemoryCacheManager.getCacheSizeAsPercentage(), 0.001);

        nativeMemoryCacheManager.close();
    }

    public void testGetIndexSizeInKilobytes() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        int genericEntryWeight = 100;
        int indexEntryWeight = 20;

        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test-1", genericEntryWeight, 0);

        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        String indexName = "test-index";
        String key = "test-key";

        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            indexEntryWeight,
            null,
            key,
            indexName
        );

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext.load()).thenReturn(indexAllocation);
        when(indexEntryContext.getKey()).thenReturn(key);

        nativeMemoryCacheManager.get(indexEntryContext, true);

        assertEquals(indexEntryWeight, nativeMemoryCacheManager.getIndexSizeInKilobytes(indexName), 0.001);

        nativeMemoryCacheManager.close();
    }

    public void testGetIndexSizeAsPercentage() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        long maxWeight = nativeMemoryCacheManager.getMaxCacheSizeInKilobytes();
        int genericEntryWeight = (int) (maxWeight / 3);
        int indexEntryWeight = (int) (maxWeight / 3);

        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test-1", genericEntryWeight, 0);

        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        String indexName = "test-index";
        String key = "test-key";

        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            indexEntryWeight,
            null,
            key,
            indexName
        );

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext.load()).thenReturn(indexAllocation);
        when(indexEntryContext.getKey()).thenReturn(key);

        nativeMemoryCacheManager.get(indexEntryContext, true);

        assertEquals(
            100 * (float) indexEntryWeight / (float) maxWeight,
            nativeMemoryCacheManager.getIndexSizeAsPercentage(indexName),
            0.001
        );

        nativeMemoryCacheManager.close();
    }

    public void testGetTrainingSize() throws ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        long maxWeight = nativeMemoryCacheManager.getMaxCacheSizeInKilobytes();
        int genericEntryWeight = (int) (maxWeight / 3);
        int allocationEntryWeight = (int) (maxWeight / 3);

        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test-1", genericEntryWeight, 0);

        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        String indexName = "test-index";
        String key = "test-key";

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            0,
            allocationEntryWeight,
            VectorDataType.FLOAT
        );

        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = mock(
            NativeMemoryEntryContext.TrainingDataEntryContext.class
        );
        when(trainingDataEntryContext.load()).thenReturn(trainingDataAllocation);
        when(trainingDataEntryContext.getKey()).thenReturn(key);

        nativeMemoryCacheManager.get(trainingDataEntryContext, true);

        assertEquals(allocationEntryWeight, nativeMemoryCacheManager.getTrainingSizeInKilobytes(), 1e-3);
        assertEquals(
            100 * (float) allocationEntryWeight / (float) maxWeight,
            nativeMemoryCacheManager.getTrainingSizeAsPercentage(),
            0.001
        );

        nativeMemoryCacheManager.close();
    }

    public void testGetIndexGraphCount() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        long maxWeight = nativeMemoryCacheManager.getMaxCacheSizeInKilobytes();
        int genericEntryWeight = (int) (maxWeight / 3);
        int indexEntryWeight = (int) (maxWeight / 3);

        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test-1", genericEntryWeight, 0);

        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        String indexName1 = "test-index-1";
        String indexName2 = "test-index-2";
        String key1 = "test-key-1";
        String key2 = "test-key-2";
        String key3 = "test-key-3";

        NativeMemoryAllocation.IndexAllocation indexAllocation1 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            indexEntryWeight,
            null,
            key1,
            indexName1
        );

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext.load()).thenReturn(indexAllocation1);
        when(indexEntryContext.getKey()).thenReturn(key1);

        nativeMemoryCacheManager.get(indexEntryContext, true);

        NativeMemoryAllocation.IndexAllocation indexAllocation2 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            indexEntryWeight,
            null,
            key2,
            indexName1
        );

        indexEntryContext = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext.load()).thenReturn(indexAllocation2);
        when(indexEntryContext.getKey()).thenReturn(key2);

        nativeMemoryCacheManager.get(indexEntryContext, true);

        NativeMemoryAllocation.IndexAllocation indexAllocation3 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            indexEntryWeight,
            null,
            key3,
            indexName2
        );

        indexEntryContext = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext.load()).thenReturn(indexAllocation3);
        when(indexEntryContext.getKey()).thenReturn(key3);

        nativeMemoryCacheManager.get(indexEntryContext, true);

        assertEquals(2, nativeMemoryCacheManager.getIndexGraphCount(indexName1));
        assertEquals(1, nativeMemoryCacheManager.getIndexGraphCount(indexName2));

        nativeMemoryCacheManager.close();
    }

    public void testGetMaxCacheSizeInKB() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        assertEquals(KNNSettings.getClusterCbLimit().getKb(), nativeMemoryCacheManager.getMaxCacheSizeInKilobytes());
        nativeMemoryCacheManager.close();
    }

    public void testGetCacheStats() throws ExecutionException {
        // Add a couple entries - confirm misses. Get a couple entries again. Confirm hits
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        String key1 = "test-1";
        String key2 = "test-2";
        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent(key1, 1);
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent(key2, 1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, true);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, true);

        CacheStats cacheStats = nativeMemoryCacheManager.getCacheStats();
        assertEquals(2, cacheStats.hitCount());
        assertEquals(2, cacheStats.missCount());
        nativeMemoryCacheManager.close();
    }

    public void testGet_evictable() throws ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        int size = 12;
        long pointer = 64;

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent("test-1", size, pointer);

        NativeMemoryAllocation testNativeMemoryAllocation = nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        assertEquals(size, nativeMemoryCacheManager.getCacheSizeInKilobytes());
        assertEquals(size, testNativeMemoryAllocation.getSizeInKB());
        assertEquals(pointer, testNativeMemoryAllocation.getMemoryAddress());

        nativeMemoryCacheManager.close();
    }

    public void testGet_unevictable() throws ExecutionException {
        // So, I think we will need to first load an entry that has a large size
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        int maxWeight = (int) nativeMemoryCacheManager.getMaxCacheSizeInKilobytes();

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent("test-1", maxWeight / 2);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);

        // Then, add another entry that would overflow the cache
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent("test-2", maxWeight);
        expectThrows(OutOfNativeMemoryException.class, () -> nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, false));
        nativeMemoryCacheManager.close();
    }

    public void testInvalidate() throws ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        String key = "test-1";

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent(key, 1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);

        long cacheSize = nativeMemoryCacheManager.getCacheSizeInKilobytes();
        assertTrue(cacheSize > 0);
        nativeMemoryCacheManager.invalidate(key);
        cacheSize = nativeMemoryCacheManager.getCacheSizeInKilobytes();
        assertEquals(0, cacheSize);

        nativeMemoryCacheManager.close();
    }

    public void testInvalidateAll() throws ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        String key1 = "test-1";
        String key2 = "test-2";
        String key3 = "test-3";

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent(key1, 1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent(key2, 1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, true);
        TestNativeMemoryEntryContent testNativeMemoryEntryContent3 = new TestNativeMemoryEntryContent(key3, 1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent3, true);

        long cacheSize = nativeMemoryCacheManager.getCacheSizeInKilobytes();
        assertTrue(cacheSize > 0);
        nativeMemoryCacheManager.invalidateAll();
        cacheSize = nativeMemoryCacheManager.getCacheSizeInKilobytes();
        assertEquals(0, cacheSize);

        nativeMemoryCacheManager.close();
    }

    public void testCacheCapacity() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        assertFalse(nativeMemoryCacheManager.isCacheCapacityReached());

        nativeMemoryCacheManager.setCacheCapacityReached(true);
        assertTrue(nativeMemoryCacheManager.isCacheCapacityReached());

        nativeMemoryCacheManager.setCacheCapacityReached(false);
        assertFalse(nativeMemoryCacheManager.isCacheCapacityReached());
        nativeMemoryCacheManager.close();
    }

    public void testGetIndicesCacheStats() throws IOException, ExecutionException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        Map<String, Map<String, Object>> indicesStats = nativeMemoryCacheManager.getIndicesCacheStats();
        assertTrue(indicesStats.isEmpty());

        // Setup 4 entries: 2 for each index
        String indexName1 = "test-index-1";
        String indexName2 = "test-index-2";

        String testKey1 = "test-1";
        String testKey2 = "test-2";
        String testKey3 = "test-3";
        String testKey4 = "test-4";

        int size1 = 3;
        int size2 = 5;

        NativeMemoryAllocation.IndexAllocation indexAllocation1 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size1,
            null,
            testKey1,
            indexName1
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation2 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size2,
            null,
            testKey2,
            indexName1
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation3 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size1,
            null,
            testKey3,
            indexName2
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation4 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size2,
            null,
            testKey4,
            indexName2
        );

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext1 = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext1.load()).thenReturn(indexAllocation1);
        when(indexEntryContext1.getKey()).thenReturn(testKey1);

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext2 = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext2.load()).thenReturn(indexAllocation2);
        when(indexEntryContext2.getKey()).thenReturn(testKey2);

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext3 = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext3.load()).thenReturn(indexAllocation3);
        when(indexEntryContext3.getKey()).thenReturn(testKey3);

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext4 = mock(NativeMemoryEntryContext.IndexEntryContext.class);
        when(indexEntryContext4.load()).thenReturn(indexAllocation4);
        when(indexEntryContext4.getKey()).thenReturn(testKey4);

        // load entries and check values returned
        nativeMemoryCacheManager.get(indexEntryContext1, true);
        nativeMemoryCacheManager.get(indexEntryContext2, true);
        nativeMemoryCacheManager.get(indexEntryContext3, true);
        nativeMemoryCacheManager.get(indexEntryContext4, true);

        indicesStats = nativeMemoryCacheManager.getIndicesCacheStats();
        assertEquals(2, indicesStats.get(indexName1).get(GRAPH_COUNT));
        assertEquals(2, indicesStats.get(indexName2).get(GRAPH_COUNT));
        assertEquals((long) (size1 + size2), indicesStats.get(indexName1).get(GRAPH_MEMORY_USAGE.getName()));
        assertEquals((long) size1 + size2, indicesStats.get(indexName2).get(GRAPH_MEMORY_USAGE.getName()));

        nativeMemoryCacheManager.close();
    }

    public void testMaintenanceScheduled() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        Cancellable maintenanceTask = nativeMemoryCacheManager.getMaintenanceTask();

        assertNotNull(maintenanceTask);

        nativeMemoryCacheManager.close();
        assertTrue(maintenanceTask.isCancelled());
    }

    @Test
    public void checkFeatureFlag() {
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterSettings.get(KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(true);
        assertTrue(KNNFeatureFlags.isForceEvictCacheEnabled());
        when(clusterSettings.get(KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(false);
        assertFalse(KNNFeatureFlags.isForceEvictCacheEnabled());
    }

    @SneakyThrows
    @Test
    public void testGet() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        Map<String, Map<String, Object>> indicesStats = nativeMemoryCacheManager.getIndicesCacheStats();
        assertTrue(indicesStats.isEmpty());

        String indexName1 = "test-index-1";
        String testKey1 = "test-1";
        int size1 = 3;
        NativeMemoryAllocation.IndexAllocation indexAllocation1 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size1,
            null,
            testKey1,
            indexName1
        );

        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext1 = spy(
            new NativeMemoryEntryContext.IndexEntryContext(
                (Directory) null,
                TestUtils.createFakeNativeMamoryCacheKey("test"),
                indexLoadStrategy,
                null,
                "test",
                knnVectorValues
            )
        );

        doReturn(indexAllocation1).when(indexEntryContext1).load();

        doReturn(0).when(indexEntryContext1).calculateSizeInKB();
        Directory mockDirectory = mock(Directory.class);
        IndexInput mockReadStream = mock(IndexInput.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockReadStream);
        // Add this line to handle the fileLength call
        when(mockDirectory.fileLength(any())).thenReturn(1024L); // 1KB for testing
        doReturn(mockDirectory).when(indexEntryContext1).getDirectory();
        assertFalse(indexEntryContext1.isIndexGraphFileOpened());
        assertEquals(indexAllocation1, nativeMemoryCacheManager.get(indexEntryContext1, false));
        // try-with-resources will anyway close the resources opened by indexEntryContext1
        assertFalse(indexEntryContext1.isIndexGraphFileOpened());
        assertEquals(indexAllocation1, nativeMemoryCacheManager.get(indexEntryContext1, false));

        verify(mockDirectory, times(1)).openInput(any(), any());
        verify(mockReadStream, times(1)).seek(0);
        verify(mockReadStream, times(2)).close();

    }

    @SneakyThrows
    @Test(expected = NullPointerException.class)
    public void testGetWithInvalidFile_NullPointerException() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = spy(
            new NativeMemoryEntryContext.IndexEntryContext(
                (Directory) null,
                "invalid-cache-key",
                indexLoadStrategy,
                null,
                "test",
                knnVectorValues
            )
        );

        Directory mockDirectory = mock(Directory.class);
        // This should throw the exception
        nativeMemoryCacheManager.get(indexEntryContext, false);
    }

    @SneakyThrows
    public void testGetWithInvalidFile_IllegalStateException() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = spy(
            new NativeMemoryEntryContext.IndexEntryContext(
                (Directory) null,
                "invalid-cache-key",
                indexLoadStrategy,
                null,
                "test",
                knnVectorValues
            )
        );

        doReturn(0).when(indexEntryContext).calculateSizeInKB();
        Directory mockDirectory = mock(Directory.class);
        // This should throw the exception
        Exception exception = Assert.assertThrows(
            UncheckedExecutionException.class,
            () -> nativeMemoryCacheManager.get(indexEntryContext, false)
        );
        assertTrue(exception.getCause() instanceof IllegalStateException);
    }

    @SneakyThrows
    @Test
    public void getWithForceEvictEnabled() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        clusterService = mock(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        clusterSettings = mock(ClusterSettings.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterSettings.get(KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(true);

        String testKey1 = "test-1";
        String indexName1 = "test-index-1";
        int size1 = 3;

        NativeMemoryAllocation.IndexAllocation indexAllocation1 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size1,
            null,
            testKey1,
            indexName1
        );

        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext1 = spy(
            new NativeMemoryEntryContext.IndexEntryContext(
                (Directory) null,
                TestUtils.createFakeNativeMamoryCacheKey("test"),
                indexLoadStrategy,
                null,
                "test",
                knnVectorValues
            )
        );

        doReturn(indexAllocation1).when(indexEntryContext1).load();
        doReturn(0).when(indexEntryContext1).calculateSizeInKB();
        Directory mockDirectory = mock(Directory.class);
        IndexInput mockReadStream = mock(IndexInput.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockReadStream);
        when(mockDirectory.fileLength(any())).thenReturn(1024L);
        doReturn(mockDirectory).when(indexEntryContext1).getDirectory();

        assertFalse(indexEntryContext1.isIndexGraphFileOpened());
        assertEquals(indexAllocation1, nativeMemoryCacheManager.get(indexEntryContext1, false));
        // In force evict path, the file should stay open since it's not in a try-with-resources
        assertTrue(indexEntryContext1.isIndexGraphFileOpened());

        assertEquals(indexAllocation1, nativeMemoryCacheManager.get(indexEntryContext1, false));
        assertTrue(indexEntryContext1.isIndexGraphFileOpened());

        // Should only be called once since second call is a cache hit
        verify(mockDirectory, times(1)).openInput(any(), any());
        verify(mockReadStream, times(1)).seek(0);
        // Since we're not closing in try-with-resources, close shouldn't be called
        verify(mockReadStream, never()).close();
    }

    @Test
    @SneakyThrows
    public void testConcurrentVectorIndexOpening() {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        clusterService = mock(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        clusterSettings = mock(ClusterSettings.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterSettings.get(KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(true);

        int numThreads = 5;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(numThreads);
        AtomicInteger openVectorIndexCalls = new AtomicInteger(0);

        // Create test allocation
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            3,
            null,
            "test-1",
            "test-index-1"
        );

        // Create and set up the spy context that will be shared across threads
        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext sharedContext = spy(
            new NativeMemoryEntryContext.IndexEntryContext(
                (Directory) null,
                TestUtils.createFakeNativeMamoryCacheKey("test"),
                indexLoadStrategy,
                null,
                "test",
                knnVectorValues
            )
        );

        // Set up mocks
        doReturn(indexAllocation).when(sharedContext).load();
        doReturn(0).when(sharedContext).calculateSizeInKB();
        Directory mockDirectory = mock(Directory.class);
        IndexInput mockReadStream = mock(IndexInput.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockReadStream);
        when(mockDirectory.fileLength(any())).thenReturn(1024L);
        doReturn(mockDirectory).when(sharedContext).getDirectory();

        // Add a delay in open to make concurrent access more likely
        doAnswer(invocation -> {
            openVectorIndexCalls.incrementAndGet();
            // Add a small delay to simulate work
            Thread.sleep(1000);
            return invocation.callRealMethod();
        }).when(sharedContext).open();

        // Create threads that will try to get the same context concurrently
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < numThreads; i++) {
            Thread t = new Thread(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    nativeMemoryCacheManager.get(sharedContext, false);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    completionLatch.countDown();
                }
            });
            threads.add(t);
            t.start();
        }

        startLatch.countDown();

        // Wait for all threads to complete
        completionLatch.await();

        // open is called for each of the threads
        verify(sharedContext, times(numThreads)).open();
        assertEquals(numThreads, openVectorIndexCalls.get());

        // but opening of the indexInput and seek only happens once, since rest of the threads will wait for first
        // thread and then pick up from cache
        verify(mockDirectory, times(1)).openInput(any(), any());
        verify(mockReadStream, times(1)).seek(0);

    }

    private static class TestNativeMemoryAllocation implements NativeMemoryAllocation {

        int size;
        long memoryAddress;

        TestNativeMemoryAllocation(int size) {
            this.size = size;
            this.memoryAddress = 0;
        }

        TestNativeMemoryAllocation(int size, long memoryAddress) {
            this.size = size;
            this.memoryAddress = memoryAddress;
        }

        @Override
        public void close() {

        }

        @Override
        public boolean isClosed() {
            return false;
        }

        @Override
        public long getMemoryAddress() {
            return memoryAddress;
        }

        @Override
        public void readLock() {

        }

        @Override
        public void writeLock() {

        }

        @Override
        public void readUnlock() {

        }

        @Override
        public void writeUnlock() {

        }

        @Override
        public int getSizeInKB() {
            return size;
        }
    }

    private static class TestNativeMemoryEntryContent extends NativeMemoryEntryContext<TestNativeMemoryAllocation> {

        long memoryAddress;
        int size;

        TestNativeMemoryEntryContent(String key, int size) {
            super(key);
            this.size = size;
            this.memoryAddress = 0;
        }

        TestNativeMemoryEntryContent(String key, int size, long memoryAddress) {
            super(key);
            this.size = size;
            this.memoryAddress = memoryAddress;
        }

        @Override
        public Integer calculateSizeInKB() {
            return size;
        }

        @Override
        public void open() {}

        @Override
        public TestNativeMemoryAllocation load() throws IOException {
            return new TestNativeMemoryAllocation(size, memoryAddress);
        }
    }
}
