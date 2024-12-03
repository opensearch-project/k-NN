/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import org.junit.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

public class CacheMaintainerTests {
    @Test
    public void testCacheEviction() throws InterruptedException {
        Cache<String, String> testCache = CacheBuilder.newBuilder().expireAfterWrite(1, TimeUnit.SECONDS).build();

        CacheMaintainer<String, String> cleaner = new CacheMaintainer<>(testCache);

        testCache.put("key1", "value1");
        assertEquals(testCache.size(), 1);

        Thread.sleep(1500);

        cleaner.cleanCache();
        assertEquals(testCache.size(), 0);

        cleaner.close();
    }
}
