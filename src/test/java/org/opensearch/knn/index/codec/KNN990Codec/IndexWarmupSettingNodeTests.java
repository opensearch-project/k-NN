/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.GRAPH_COUNT;

public class IndexWarmupSettingNodeTests extends KNNSingleNodeTestCase {
    private final String testIndexName1 = "test-index1";
    private final String testIndexName2 = "test-index2";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    public void testWarmup_onSegmentCreated() throws IOException, ExecutionException, InterruptedException {
        createIndex(testIndexName1, getKNNDefaultIndexSettingsBuildsGraphAlways().build());
        createKnnIndexMapping(testIndexName1, testFieldName, dimensions);
        addKnnDoc(testIndexName1, "1", testFieldName, new Float[] { randomFloat(), randomFloat() });
        assertNull(NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName1));

        Settings.Builder builder = getKNNDefaultIndexSettingsBuildsGraphAlways();
        setIndexWarmupEnabledSetting(builder, true);
        createIndex(testIndexName2, builder.build());
        createKnnIndexMapping(testIndexName2, testFieldName, dimensions);
        addKnnDoc(testIndexName2, "1", testFieldName, new Float[] { randomFloat(), randomFloat() });
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName2).get(GRAPH_COUNT));

        addKnnDoc(testIndexName2, "2", testFieldName, new Float[] { randomFloat(), randomFloat() });
        assertEquals(2, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName2).get(GRAPH_COUNT));
    }

    public void testWarmup_onMemoryOptimizedFields() throws IOException, ExecutionException, InterruptedException {
        Settings.Builder builder = getKNNDefaultIndexSettingsBuildsGraphAlways();
        setIndexWarmupEnabledSetting(builder, true);
        setMemoryOptimizedSearchEnabled(builder, true);
        createIndex(testIndexName1, builder.build());
        createKnnIndexMapping(testIndexName1, testFieldName, dimensions);
        addKnnDoc(testIndexName1, "1", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName1, "2", testFieldName, new Float[] { randomFloat(), randomFloat() });
        assertNull(NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName1));
    }

    public void testWarmup_onMerge() throws IOException, ExecutionException, InterruptedException {
        createIndex(testIndexName1, getKNNDefaultIndexSettingsBuildsGraphAlways().build());
        createKnnIndexMapping(testIndexName1, testFieldName, dimensions);
        addKnnDoc(testIndexName1, "1", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName1, "2", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName1, "3", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName1, "4", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName1, "5", testFieldName, new Float[] { randomFloat(), randomFloat() });
        mergeIndex(testIndexName1, 1);
        assertNull(NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName1));

        Settings.Builder builder = getKNNDefaultIndexSettingsBuildsGraphAlways();
        setIndexWarmupEnabledSetting(builder, true);
        createIndex(testIndexName2, builder.build());
        createKnnIndexMapping(testIndexName2, testFieldName, dimensions);
        addKnnDoc(testIndexName2, "1", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName2, "2", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName2, "3", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName2, "4", testFieldName, new Float[] { randomFloat(), randomFloat() });
        addKnnDoc(testIndexName2, "5", testFieldName, new Float[] { randomFloat(), randomFloat() });
        mergeIndex(testIndexName2, 1);
        assertEquals(6, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().get(testIndexName2).get(GRAPH_COUNT));
    }
}
