/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.transport.grpc.spi.QueryBuilderProtoConverterRegistry;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.KnnQueryRescore;
import org.opensearch.protobufs.ObjectMap;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

public class KNNQueryBuilderProtoUtilsTests extends OpenSearchTestCase {

    @Mock
    private QueryBuilderProtoConverterRegistry mockRegistry;

    @Mock
    private QueryBuilder mockQueryBuilder;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        // Set up the registry for each test
        KNNQueryBuilderProtoUtils.setRegistry(mockRegistry);
    }

    @Test
    public void testFromProto_basicFields() {
        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").addVector(1.0f).addVector(2.0f).addVector(3.0f).setK(5).build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals("test_field", knnQueryBuilder.fieldName());
        assertArrayEquals(new float[] { 1.0f, 2.0f, 3.0f }, (float[]) knnQueryBuilder.vector(), 0.001f);
        assertEquals(5, knnQueryBuilder.getK());
    }

    @Test
    public void testFromProto_withBoost() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setBoost(2.5f)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(2.5f, knnQueryBuilder.boost(), 0.001f);
    }

    @Test
    public void testFromProto_withMaxDistance() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setMaxDistance(0.75f)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(0.75f, knnQueryBuilder.getMaxDistance(), 0.001f);
    }

    @Test
    public void testFromProto_withMinScore() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setMinScore(0.85f)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(0.85f, knnQueryBuilder.getMinScore(), 0.001f);
    }

    @Test
    public void testFromProto_withQueryName() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setUnderscoreName("test_query")
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals("test_query", knnQueryBuilder.queryName());
    }

    @Test
    public void testFromProto_withExpandNested() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setExpandNestedDocs(true)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertTrue(knnQueryBuilder.getExpandNested());
    }

    @Test
    public void testFromProto_withMethodParameters() {
        ObjectMap.Value efSearchValue = ObjectMap.Value.newBuilder().setInt32(100).build();
        ObjectMap.Value nprobesValue = ObjectMap.Value.newBuilder().setInt32(10).build();
        ObjectMap methodParams = ObjectMap.newBuilder().putFields("ef_search", efSearchValue).putFields("nprobes", nprobesValue).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setMethodParameters(methodParams)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        Map<String, ?> methodParameters = knnQueryBuilder.getMethodParameters();
        assertNotNull(methodParameters);
        assertEquals(2, methodParameters.size());
        assertEquals(100, methodParameters.get("ef_search"));
        assertEquals(10, methodParameters.get("nprobes"));
    }

    @Test
    public void testFromProto_withFilter() {
        QueryContainer filterContainer = QueryContainer.newBuilder().build();
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setFilter(filterContainer)
            .build();

        when(mockRegistry.fromProto(any())).thenReturn(mockQueryBuilder);

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getFilter());
        assertEquals(mockQueryBuilder, knnQueryBuilder.getFilter());
    }

    @Test
    public void testFromProto_withRescoreEnable() {
        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setEnable(true).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getRescoreContext());
        assertEquals(RescoreContext.getDefault(), knnQueryBuilder.getRescoreContext());
    }

    @Test
    public void testFromProto_withRescoreDisable() {
        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setEnable(false).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getRescoreContext());
        assertEquals(RescoreContext.EXPLICITLY_DISABLED_RESCORE_CONTEXT, knnQueryBuilder.getRescoreContext());
    }

    @Test
    public void testFromProto_withRescoreContext() {
        org.opensearch.protobufs.RescoreContext rescoreContext = org.opensearch.protobufs.RescoreContext.newBuilder()
            .setOversampleFactor(3.5f)
            .build();

        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setContext(rescoreContext).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getRescoreContext());
        assertEquals(3.5f, knnQueryBuilder.getRescoreContext().getOversampleFactor(), 0.001f);
    }

    @Test
    public void testFromProto_withRescoreContextNoOversampleFactor() {
        org.opensearch.protobufs.RescoreContext rescoreContext = org.opensearch.protobufs.RescoreContext.newBuilder().build();

        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setContext(rescoreContext).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getRescoreContext());
        assertEquals(RescoreContext.getDefault(), knnQueryBuilder.getRescoreContext());
    }

    @Test
    public void testFromProto_withNullRegistry_withFilter() {
        // Test that queries with filters fail with null registry
        KNNQueryBuilderProtoUtils.setRegistry(null);

        QueryContainer filterContainer = QueryContainer.newBuilder().build();
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setFilter(filterContainer)
            .build();

        // Should throw NPE when trying to use null registry
        expectThrows(NullPointerException.class, () -> { KNNQueryBuilderProtoUtils.fromProto(knnQuery); });
    }

    @Test
    public void testFromProto_invalidMethodParameter() {
        // Test with invalid method parameter
        ObjectMap.Value invalidValue = ObjectMap.Value.newBuilder().setString("invalid_ef_search").build();
        ObjectMap methodParams = ObjectMap.newBuilder().putFields("invalid_param_name", invalidValue).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setMethodParameters(methodParams)
            .build();

        // Should throw IllegalArgumentException for unknown method parameter
        expectThrows(IllegalArgumentException.class, () -> { KNNQueryBuilderProtoUtils.fromProto(knnQuery); });
    }
}
