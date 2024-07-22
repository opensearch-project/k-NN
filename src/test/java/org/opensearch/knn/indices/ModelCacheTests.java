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
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.MODEL_CACHE_SIZE_LIMIT_SETTING;
import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;

public class ModelCacheTests extends KNNTestCase {

    public void testGet_normal() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;
        Model mockModel = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            "hello".getBytes(),
            modelId
        );
        String cacheSize = "10%";

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
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
        String cacheSize = "1kb";

        Model mockModel = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[BYTES_PER_KILOBYTES + 1],
            modelId
        );

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
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
        String cacheSize = "10%";

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenThrow(new IllegalArgumentException());

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
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
        String cacheSize = "10%";

        int size1 = BYTES_PER_KILOBYTES;
        Model mockModel1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[size1],
            modelId1
        );
        int size2 = BYTES_PER_KILOBYTES * 3;
        Model mockModel2 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[size2],
            modelId2
        );

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);
        when(modelDao.get(modelId2)).thenReturn(mockModel2);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        modelCache.get(modelId1);
        modelCache.get(modelId2);
        modelCache.get(modelId1);
        modelCache.get(modelId2);

        assertEquals((size1 + size2) / BYTES_PER_KILOBYTES + 2, modelCache.getTotalWeightInKB());
    }

    public void testRemove_normal() throws ExecutionException, InterruptedException {
        String modelId1 = "test-model-id-1";
        String modelId2 = "test-model-id-2";
        int dimension = 2;
        String cacheSize = "10%";

        int size1 = BYTES_PER_KILOBYTES;
        Model mockModel1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[size1],
            modelId1
        );
        int size2 = BYTES_PER_KILOBYTES * 3;
        Model mockModel2 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[size2],
            modelId2
        );

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);
        when(modelDao.get(modelId2)).thenReturn(mockModel2);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        modelCache.get(modelId1);
        modelCache.get(modelId2);
        modelCache.get(modelId1);
        modelCache.get(modelId2);

        assertEquals(((size1 + size2) / BYTES_PER_KILOBYTES) + 2, modelCache.getTotalWeightInKB());

        modelCache.remove(modelId1);

        assertEquals((size2 / BYTES_PER_KILOBYTES) + 1, modelCache.getTotalWeightInKB());

        modelCache.remove(modelId2);

        assertEquals(0, modelCache.getTotalWeightInKB());
    }

    public void testRebuild_normal() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;
        String cacheSize = "10%";
        Model mockModel = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            "hello".getBytes(),
            modelId
        );

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        // Add element to cache - nothing should be kept
        modelCache.get(modelId);
        assertEquals((mockModel.getModelBlob().length / BYTES_PER_KILOBYTES) + 1, modelCache.getTotalWeightInKB());

        // Rebuild and make sure cache is empty
        modelCache.rebuild();
        assertEquals(0, modelCache.getTotalWeightInKB());

        // Add element again
        modelCache.get(modelId);
        assertEquals((mockModel.getModelBlob().length / BYTES_PER_KILOBYTES) + 1, modelCache.getTotalWeightInKB());
    }

    public void testRebuild_afterSettingUpdate() throws ExecutionException, InterruptedException {
        String modelId = "test-model-id";
        int dimension = 2;

        int modelSize = 2 * BYTES_PER_KILOBYTES;
        Model mockModel = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[modelSize],
            modelId
        );

        String cacheSize1 = "1kb";
        String cacheSize2 = "4kb";

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize1).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        // Add element to cache - element should not remain in cache
        modelCache.get(modelId);
        assertEquals(0, modelCache.getTotalWeightInKB());

        // Rebuild and make sure cache is empty
        Settings newSettings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize2).build();
        clusterService.getClusterSettings().applySettings(newSettings);
        assertEquals(0, modelCache.getTotalWeightInKB());

        // Add element again - element should remain in cache
        modelCache.get(modelId);
        assertEquals((modelSize / BYTES_PER_KILOBYTES) + 1, modelCache.getTotalWeightInKB());
    }

    public void testRemove_modelNotInCache() {
        String modelId1 = "test-model-id-1";
        String cacheSize = "10%";

        ModelDao modelDao = mock(ModelDao.class);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        assertEquals(0, modelCache.getTotalWeightInKB());
        modelCache.remove(modelId1);
        assertEquals(0, modelCache.getTotalWeightInKB());
    }

    public void testContains() throws ExecutionException, InterruptedException {
        String modelId1 = "test-model-id-1";
        int dimension = 2;
        int modelSize1 = 100;
        Model mockModel1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[modelSize1],
            modelId1
        );

        String modelId2 = "test-model-id-2";

        String cacheSize = "10%";

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
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
        int modelSize1 = BYTES_PER_KILOBYTES;
        Model mockModel1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[modelSize1],
            modelId1
        );

        String modelId2 = "test-model-id-2";
        int modelSize2 = BYTES_PER_KILOBYTES * 2;
        Model mockModel2 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[modelSize2],
            modelId2
        );

        String cacheSize = "10%";

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId1)).thenReturn(mockModel1);
        when(modelDao.get(modelId2)).thenReturn(mockModel2);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();

        modelCache.get(modelId1);
        modelCache.get(modelId2);

        assertEquals(((modelSize1 + modelSize2) / BYTES_PER_KILOBYTES) + 2, modelCache.getTotalWeightInKB());
        modelCache.removeAll();
        assertEquals(0, modelCache.getTotalWeightInKB());
    }

    public void testModelCacheEvictionDueToSize() throws ExecutionException, InterruptedException {
        String modelIdPattern = "test-model-id-%d";
        int dimension = 2;
        int maxDocuments = 10;
        ModelDao modelDao = mock(ModelDao.class);
        for (int i = 0; i < maxDocuments; i++) {
            String modelId = String.format(modelIdPattern, i);
            Model mockModel = new Model(
                new ModelMetadata(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    dimension,
                    ModelState.CREATED,
                    ZonedDateTime.now(ZoneOffset.UTC).toString(),
                    "",
                    "",
                    "",
                    MethodComponentContext.EMPTY,
                    VectorDataType.DEFAULT
                ),
                new byte[BYTES_PER_KILOBYTES * 2],
                modelId
            );
            when(modelDao.get(modelId)).thenReturn(mockModel);
        }

        String cacheSize = "10kb";
        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelCache.initialize(modelDao, clusterService);
        ModelCache modelCache = new ModelCache();
        assertNull(modelCache.getEvictedDueToSizeAt());
        for (int i = 0; i < maxDocuments; i++) {
            modelCache.get(String.format(modelIdPattern, i));
        }
        assertNotNull(modelCache.getEvictedDueToSizeAt());
    }
}
