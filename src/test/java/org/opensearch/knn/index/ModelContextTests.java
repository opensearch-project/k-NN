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

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.UncheckedExecutionException;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;

import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.MODEL_CACHE_SIZE_IN_BYTES_SETTING;

public class ModelContextTests extends KNNTestCase {
    public void testGetModelId() {
        String modelId = "test-model-id";
        ModelContext modelContext = new ModelContext(modelId, KNNEngine.DEFAULT, SpaceType.DEFAULT, 2);
        assertEquals(modelId, modelContext.getModelId());
    }

    public void testGetKNNEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelContext modelContext = new ModelContext(null, knnEngine, SpaceType.DEFAULT, 2);
        assertEquals(knnEngine, modelContext.getKNNEngine());
    }

    public void testGetSpaceType() {
        SpaceType spaceType = SpaceType.DEFAULT;
        ModelContext modelContext = new ModelContext(null, KNNEngine.DEFAULT, spaceType, 2);
        assertEquals(spaceType, modelContext.getSpaceType());
    }

    public void testGetDimension() {
        int dimension = 17;
        ModelContext modelContext = new ModelContext(null, KNNEngine.DEFAULT, SpaceType.DEFAULT, dimension);
        assertEquals(dimension, modelContext.getDimension());
    }

    public void testParse() throws ExecutionException, InterruptedException {
        Object invalidModelId = 15;
        expectThrows(MapperParsingException.class, () -> ModelContext.parse(invalidModelId));

        ModelDao modelDao = mock(ModelDao.class);

        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_IN_BYTES_SETTING.getKey(), 10).build();
        ClusterSettings clusterSettings = new ClusterSettings(settings,
                ImmutableSet.of(MODEL_CACHE_SIZE_IN_BYTES_SETTING));

        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getSettings()).thenReturn(settings);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);

        ModelCache.getInstance().removeAll();
        ModelCache.initialize(modelDao, clusterService);

        String nonexistantModelId = "nonexistant-model";

        doThrow(IllegalArgumentException.class).when(modelDao).get(nonexistantModelId);
        expectThrows(UncheckedExecutionException.class, () -> ModelContext.parse(nonexistantModelId));

        String modelId = "test-model";
        Model mockModel = new Model(KNNEngine.DEFAULT, SpaceType.DEFAULT, 2, new byte[2]);
        when(modelDao.get(modelId)).thenReturn(mockModel);

        ModelContext modelContext = ModelContext.parse(modelId);
        assertEquals(modelId, modelContext.getModelId());
        assertEquals(mockModel.getKnnEngine(), modelContext.getKNNEngine());
        assertEquals(mockModel.getSpaceType(), modelContext.getSpaceType());
        assertEquals(mockModel.getDimension(), modelContext.getDimension());
    }
}
