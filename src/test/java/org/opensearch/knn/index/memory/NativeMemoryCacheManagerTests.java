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
import org.opensearch.knn.common.exception.NativeMemoryThrottleException;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class NativeMemoryCacheManagerTests extends OpenSearchSingleNodeTestCase {

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return Collections.singletonList(KNNPlugin.class);
    }

    public void testRebuildCache() throws ExecutionException, InterruptedException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        // Put entry in cache and check that the weight matches
        long size = 10;
        TestNativeMemoryEntryContent testNativeMemoryEntryContent = new TestNativeMemoryEntryContent("test", size);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent, true);

        assertEquals(size, nativeMemoryCacheManager.getCacheWeight());

        // Call rebuild and check total weight is at 0
        nativeMemoryCacheManager.rebuildCache();

        // Sleep for a second or two so that the executor can invalidate all entries
        Thread.sleep(2000);

        assertEquals(0, nativeMemoryCacheManager.getCacheWeight());
        nativeMemoryCacheManager.close();
    }

    public void testGetCacheWeight() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        assertEquals(0, nativeMemoryCacheManager.getCacheWeight());

        // Put 2 entries in cache and check that the weight matches
        long size1 = 10;
        long size2 = 20;
        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent("test-1", size1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent("test-2", size2);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, true);

        assertEquals(size1 + size2, nativeMemoryCacheManager.getCacheWeight());
        nativeMemoryCacheManager.close();
    }


    public void testGetCacheMaxWeight() throws IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        assertEquals(KNNSettings.getCircuitBreakerLimit().getKb(), nativeMemoryCacheManager.getCacheMaxWeight());
        nativeMemoryCacheManager.close();
    }

    public void testGetCacheStats() throws ExecutionException, IOException {
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

    public void testGetCacheAsMap() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        String key1 = "test-1";
        String key2 = "test-2";
        long size1 = 12;
        long size2 = 13;

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent(key1, size1);
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent(key2, size2);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, true);

        Map<String, NativeMemoryAllocation> cacheAsMap = nativeMemoryCacheManager.getCacheAsMap();

        assertTrue(cacheAsMap.containsKey(key1));
        assertTrue(cacheAsMap.containsKey(key2));

        assertEquals(size1, cacheAsMap.get(key1).getSize());
        assertEquals(size2, cacheAsMap.get(key2).getSize());

        nativeMemoryCacheManager.close();
    }

    public void testGet_evictable() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        long size = 12;
        long pointer = 64;

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent("test-1", size, pointer);

        NativeMemoryAllocation testNativeMemoryAllocation = nativeMemoryCacheManager.get(testNativeMemoryEntryContent1,
                true);
        assertEquals(size, nativeMemoryCacheManager.getCacheWeight());
        assertEquals(size, testNativeMemoryAllocation.getSize());
        assertEquals(pointer, testNativeMemoryAllocation.getPointer());

        nativeMemoryCacheManager.close();
    }

    public void testGet_unevictable() throws ExecutionException, IOException {
        // So, I think we will need to first load an entry that has a large size
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();
        long maxWeight = nativeMemoryCacheManager.getCacheMaxWeight();

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent("test-1", maxWeight/2);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);

        // Then, add another entry that would overflow the cache
        TestNativeMemoryEntryContent testNativeMemoryEntryContent2 = new TestNativeMemoryEntryContent("test-2", maxWeight);
        expectThrows(NativeMemoryThrottleException.class, () ->nativeMemoryCacheManager.get(testNativeMemoryEntryContent2, false));
        nativeMemoryCacheManager.close();
    }

    public void testInvalidate() throws ExecutionException, IOException {
        NativeMemoryCacheManager nativeMemoryCacheManager = new NativeMemoryCacheManager();

        String key = "test-1";

        TestNativeMemoryEntryContent testNativeMemoryEntryContent1 = new TestNativeMemoryEntryContent(key, 1);
        nativeMemoryCacheManager.get(testNativeMemoryEntryContent1, true);

        Map<String, NativeMemoryAllocation> cacheAsMap = nativeMemoryCacheManager.getCacheAsMap();
        assertEquals(1, cacheAsMap.size());
        nativeMemoryCacheManager.invalidate(key);
        assertEquals(0, cacheAsMap.size());

        nativeMemoryCacheManager.close();
    }

    public void testInvalidateAll() throws IOException, ExecutionException {
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

        Map<String, NativeMemoryAllocation> cacheAsMap = nativeMemoryCacheManager.getCacheAsMap();
        assertEquals(3, cacheAsMap.size());
        nativeMemoryCacheManager.invalidateAll();
        assertEquals(0, cacheAsMap.size());

        nativeMemoryCacheManager.close();
    }

    private static class TestNativeMemoryAllocation implements NativeMemoryAllocation {

        long size;
        long pointer;

        TestNativeMemoryAllocation(long size) {
            this.size = size;
            this.pointer = 0;
        }

        TestNativeMemoryAllocation(long size, long pointer) {
            this.size = size;
            this.pointer = pointer;
        }

        @Override
        public void close() throws InterruptedException {

        }

        @Override
        public boolean isClosed() {
            return false;
        }

        @Override
        public long getPointer() {
            return pointer;
        }

        @Override
        public void readLock() throws InterruptedException {

        }

        @Override
        public void writeLock() throws InterruptedException {

        }

        @Override
        public void readUnlock() throws InterruptedException {

        }

        @Override
        public void writeUnlock() {

        }

        @Override
        public long getSize() {
            return size;
        }
    }

    private static class TestNativeMemoryEntryContent extends NativeMemoryEntryContext<TestNativeMemoryAllocation> {

        long pointer;

        TestNativeMemoryEntryContent(String key, long size) {
            super(key, size);
            this.pointer = 0;
        }

        TestNativeMemoryEntryContent(String key, long size, long pointer) {
            super(key, size);
            this.pointer = pointer;
        }

        @Override
        public TestNativeMemoryAllocation load() throws IOException {
            return new TestNativeMemoryAllocation(size, pointer);
        }
    }
}
