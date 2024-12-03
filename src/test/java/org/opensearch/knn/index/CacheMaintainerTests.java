/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import org.junit.Test;
import org.opensearch.knn.index.util.ScheduledExecutor;

import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

public class CacheMaintainerTests {
    @Test
    public void testCacheEviction() throws InterruptedException {
        Cache<String, String> testCache = CacheBuilder.newBuilder().expireAfterWrite(1, TimeUnit.SECONDS).build();

        ScheduledExecutor executor = new ScheduledExecutor(testCache::cleanUp, 60 * 1000);

        testCache.put("key1", "value1");
        assertEquals(testCache.size(), 1);

        Thread.sleep(1500);

        executor.getTask().run();

        assertEquals(testCache.size(), 0);

        executor.close();
    }
}
