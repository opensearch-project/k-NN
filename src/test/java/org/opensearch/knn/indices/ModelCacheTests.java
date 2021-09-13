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

package org.opensearch.knn.indices;

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.UncheckedExecutionException;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.MODEL_CACHE_SIZE_IN_BYTES_SETTING;

public class ModelCacheTests extends KNNTestCase {

    public void testGet_normal() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;
        Model mockModel = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), "hello".getBytes());
        long cacheSize = 100L;

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        assertEquals(mockModel, modelCache.get(modelId));
    }

    public void testGet_modelDoesNotFitInCache() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;
        long cacheSize = 500;

        Model mockModel = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension),
                new byte[Long.valueOf(cacheSize).intValue() + 1]);

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        assertEquals(mockModel, modelCache.get(modelId));
        assertFalse(modelCache.contains(modelId));
    }

    public void testGet_modelDoesNotExist() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        long cacheSize = 100L;

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenThrow(new IllegalArgumentException());

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        expectThrows(UncheckedExecutionException.class, () -> modelCache.get(modelId));
    }

    public void testGetTotalWeight() throws ExecutionException, InterruptedException {
        String modelId1 = "test-model-id-1";
        String modelId2 = "test-model-id-2";
        int dimension = 2;
        long cacheSize = 500L;

        int size1 = 100;
        Model mockModel1 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[size1]);
        int size2 = 300;
        Model mockModel2 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[size2]);

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);
        when(modelDao.get(modelId2)).thenReturn(mockModel2);


        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        modelCache.get(modelId1);
        modelCache.get(modelId2);
        modelCache.get(modelId1);
        modelCache.get(modelId2);

        assertEquals(size1 + size2, modelCache.getTotalWeight());
    }

    public void testRemove_normal() throws ExecutionException, InterruptedException {
        String modelId1 = "test-model-id-1";
        String modelId2 = "test-model-id-2";
        int dimension = 2;
        long cacheSize = 500L;

        int size1 = 100;
        Model mockModel1 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[size1]);
        int size2 = 300;
        Model mockModel2 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[size2]);

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);
        when(modelDao.get(modelId2)).thenReturn(mockModel2);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        modelCache.get(modelId1);
        modelCache.get(modelId2);
        modelCache.get(modelId1);
        modelCache.get(modelId2);

        assertEquals(size1 + size2, modelCache.getTotalWeight());

        modelCache.remove(modelId1);

        assertEquals( size2, modelCache.getTotalWeight());

        modelCache.remove(modelId2);

        assertEquals( 0, modelCache.getTotalWeight());
    }

    public void testRebuild_normal() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;
        long cacheSize = 100L;
        Model mockModel = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), "hello".getBytes());

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        // Add element to cache - nothing should be kept
        modelCache.get(modelId);
        assertEquals(mockModel.getModelBlob().length, modelCache.getTotalWeight());

        // Rebuild and make sure cache is empty
        modelCache.rebuild();
        assertEquals(0, modelCache.getTotalWeight());

        // Add element again
        modelCache.get(modelId);
        assertEquals(mockModel.getModelBlob().length, modelCache.getTotalWeight());
    }

    public void testRebuild_afterSettingUpdate() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;

        int modelSize = 101;
        Model mockModel = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[modelSize]);

        long cacheSize1 = 100L;
        long cacheSize2 = 200L;

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize1).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        // Add element to cache - element should not remain in cache
        modelCache.get(modelId);
        assertEquals(0, modelCache.getTotalWeight());

        // Rebuild and make sure cache is empty
        Settings newSettings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize2).build();
        clusterService.getClusterSettings().applySettings(newSettings);
        assertEquals(0, modelCache.getTotalWeight());

        // Add element again - element should remain in cache
        modelCache.get(modelId);
        assertEquals(modelSize, modelCache.getTotalWeight());
    }

    public void testRemove_modelNotInCache() {
        String modelId1 = "test-model-id-1";
        long cacheSize = 100L;

        ModelDao modelDao = mock(ModelDao.class);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        assertEquals( 0, modelCache.getTotalWeight());
        modelCache.remove(modelId1);
        assertEquals( 0, modelCache.getTotalWeight());
    }

    public void testContains() throws ExecutionException, InterruptedException {
        String modelId1 = "test-model-id-1";
        int dimension = 2;
        int modelSize1 = 100;
        Model mockModel1 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[modelSize1]);

        String modelId2 = "test-model-id-2";

        long cacheSize = 500L;

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        assertFalse(modelCache.contains(modelId1));
        modelCache.get(modelId1);
        assertTrue(modelCache.contains(modelId1));
        assertFalse(modelCache.contains(modelId2));
    }

    public void testRemoveAll() throws ExecutionException, InterruptedException {
        int dimension = 2;
        String modelId1 = "test-model-id-1";
        int modelSize1 = 100;
        Model mockModel1 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[modelSize1]);

        String modelId2 = "test-model-id-2";
        int modelSize2 = 100;
        Model mockModel2 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension), new byte[modelSize2]);

        long cacheSize = 500L;

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);
        when(modelDao.get(modelId2)).thenReturn(mockModel2);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        modelCache.get(modelId1);
        modelCache.get(modelId2);

        assertEquals( modelSize1 + modelSize2, modelCache.getTotalWeight());
        modelCache.removeAll();
        assertEquals( 0, modelCache.getTotalWeight());
    }
}
