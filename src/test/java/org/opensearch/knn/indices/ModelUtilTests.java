/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.indices;

import org.junit.Assert;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;

public class ModelUtilTests extends KNNTestCase {
    private static final String MODEL_ID = "test-model";

    public void testGetModelMetadata_whenVariousInputs_thenSuccess() {
        Assert.assertNull(ModelUtil.getModelMetadata(null));
        Assert.assertNull(ModelUtil.getModelMetadata(""));

        ModelCache modelCache = Mockito.mock(ModelCache.class);
        Model model = Mockito.mock(Model.class);
        ModelMetadata modelMetadata = Mockito.mock(ModelMetadata.class);
        MockedStatic<ModelCache> modelCacheMockedStatic = Mockito.mockStatic(ModelCache.class);

        modelCacheMockedStatic.when(ModelCache::getInstance).thenReturn(modelCache);
        try (MockedStatic<ModelDao.OpenSearchKNNModelDao> modelDaoMockedStatic = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
            ModelDao.OpenSearchKNNModelDao modelDao = Mockito.mock(ModelDao.OpenSearchKNNModelDao.class);
            Mockito.when(modelDao.getMetadata(MODEL_ID)).thenReturn(modelMetadata);
            Mockito.when(modelMetadata.getState()).thenReturn(ModelState.FAILED);
            modelDaoMockedStatic.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(modelDao);

            Mockito.when(modelCache.get(MODEL_ID)).thenReturn(model);
            Mockito.when(model.getModelMetadata()).thenReturn(null);
            Assert.assertThrows(IllegalArgumentException.class, () -> ModelUtil.getModelMetadata(MODEL_ID));

            Mockito.when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
            Assert.assertNotNull(ModelUtil.getModelMetadata(MODEL_ID));
        }
        modelCacheMockedStatic.close();
    }
}
