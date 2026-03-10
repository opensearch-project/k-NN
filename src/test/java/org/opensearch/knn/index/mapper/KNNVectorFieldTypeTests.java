/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.index.mapper.ArraySourceValueFetcher;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

public class KNNVectorFieldTypeTests extends KNNTestCase {
    private static final String FIELD_NAME = "test-field";

    public void testValueFetcher() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        ValueFetcher valueFetcher = knnVectorFieldType.valueFetcher(mockQueryShardContext, null, null);
        assertTrue(valueFetcher instanceof ArraySourceValueFetcher);
    }

    public void testResolveRescoreContext_whenFlatMethod_thenReturnOversampleFactor2() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(flatMethodContext, 128)
        );
        RescoreContext rescoreContext = knnVectorFieldType.resolveRescoreContext(null);
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.001f);
        assertFalse(rescoreContext.isUserProvided());
    }

    public void testResolveRescoreContext_whenFlatMethodWithUserProvidedContext_thenReturnUserContext() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(flatMethodContext, 128)
        );
        RescoreContext userContext = RescoreContext.builder().oversampleFactor(5.0f).userProvided(true).build();
        RescoreContext rescoreContext = knnVectorFieldType.resolveRescoreContext(userContext);
        assertSame(userContext, rescoreContext);
    }
}
