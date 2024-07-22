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

package org.opensearch.knn.plugin.transport;

import com.google.common.collect.ImmutableSet;
import org.junit.Ignore;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.MODEL_CACHE_SIZE_LIMIT_SETTING;

public class RemoveModelFromCacheTransportActionTests extends KNNSingleNodeTestCase {

    @Ignore
    public void testNodeOperation_modelNotInCache() {
        ClusterService clusterService = mock(ClusterService.class);
        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), "10%").build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelDao modelDao = mock(ModelDao.class);
        ModelCache.initialize(modelDao, clusterService);

        // Check that model cache is initially empty
        ModelCache modelCache = ModelCache.getInstance();
        assertEquals(0, modelCache.getTotalWeightInKB());

        // Remove the model from the cache
        RemoveModelFromCacheTransportAction action = node().injector().getInstance(RemoveModelFromCacheTransportAction.class);

        RemoveModelFromCacheNodeRequest request = new RemoveModelFromCacheNodeRequest("invalid-model");
        action.nodeOperation(request);

        assertEquals(0L, modelCache.getTotalWeightInKB());
    }

    @Ignore
    public void testNodeOperation_modelInCache() throws ExecutionException, InterruptedException {
        ClusterService clusterService = mock(ClusterService.class);
        Settings settings = Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), "10%").build();
        ClusterSettings clusterSettings = new ClusterSettings(settings, ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING));
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        ModelDao modelDao = mock(ModelDao.class);
        String modelId = "test-model-id";
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L2,
                16,
                ModelState.CREATED,
                "timestamp",
                "description",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[128],
            modelId
        );

        when(modelDao.get(modelId)).thenReturn(model);

        ModelCache.initialize(modelDao, clusterService);

        // Load the model into the cache
        ModelCache modelCache = ModelCache.getInstance();
        modelCache.get(modelId);

        // Remove the model from the cache
        RemoveModelFromCacheTransportAction action = node().injector().getInstance(RemoveModelFromCacheTransportAction.class);

        RemoveModelFromCacheNodeRequest request = new RemoveModelFromCacheNodeRequest(modelId);
        action.nodeOperation(request);

        assertEquals(0L, modelCache.getTotalWeightInKB());
    }
}
