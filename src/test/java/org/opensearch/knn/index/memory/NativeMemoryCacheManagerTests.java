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
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.common.exception.OutOfNativeMemoryException;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.GRAPH_COUNT;
import static org.opensearch.knn.plugin.stats.StatNames.GRAPH_MEMORY_USAGE;

public class NativeMemoryCacheManagerTests extends OpenSearchSingleNodeTestCase {

    @Override
    public void tearDown() throws Exception {
        // Clear out persistent metadata
        ClusterUpdateSettingsRequest clusterUpdateSettingsRequest = new ClusterUpdateSettingsRequest();
        Settings circuitBreakerSettings = Settings.builder().putNull(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED).build();
        clusterUpdateSettingsRequest.persistentSettings(circuitBreakerSettings);
        client().admin().cluster().updateSettings(clusterUpdateSettingsRequest).get();
        super.tearDown();
    }

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return Collections.singletonList(KNNPlugin.class);
    }

    public void testRebuildCache() throws ExecutionException, InterruptedException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        // Put entry in cache and check that the weight matches
        int size = 10;
        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test", size);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        assertEquals(size, nativeMemoryCacheManager.getCacheSizeInKilobytes());

        // Call rebuild and check total weight is at 0
        nativeMemoryCacheManager.rebuildCache();

        // Sleep for a second or two so that the executor can invalidate all entries
        Thread.sleep(2000);

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
            indexName,
            null
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
            indexName,
            null
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

        assertEquals((float) allocationEntryWeight, nativeMemoryCacheManager.getTrainingSizeInKilobytes(), 0.001);
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
            indexName1,
            null
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
            indexName1,
            null
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
            indexName2,
            null
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
        assertEquals(KNNSettings.getCircuitBreakerLimit().getKb(), nativeMemoryCacheManager.getMaxCacheSizeInKilobytes());
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
            indexName1,
            null
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation2 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size2,
            null,
            testKey2,
            indexName1,
            null
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation3 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size1,
            null,
            testKey3,
            indexName2,
            null
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation4 = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size2,
            null,
            testKey4,
            indexName2,
            null
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
        public TestNativeMemoryAllocation load() throws IOException {
            return new TestNativeMemoryAllocation(size, memoryAddress);
        }
    }
}
