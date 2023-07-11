/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.common.io.stream.NamedWriteableRegistry;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.Index;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.KNNClusterUtil;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.plugins.SearchPlugin;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;

public class KNNQueryBuilderTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final int K = 1;
    private static final TermQueryBuilder TERM_QUERY = QueryBuilders.termQuery("field", "value");
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

    public void testInvalidK() {
        float[] queryVector = { 1.0f, 1.0f };

        /**
         * -ve k
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, -K));

        /**
         * zero k
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, 0));

        /**
         * k > KNNQueryBuilder.K_MAX
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, KNNQueryBuilder.K_MAX + K));
    }

    public void testEmptyVector() {
        /**
         * null query vector
         */
        float[] queryVector = null;
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, K));

        /**
         * empty query vector
         */
        float[] queryVector1 = {};
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector1, K));
    }

    public void testFromXcontent() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilder.fromXContent(contentParser);
        actualBuilder.equals(knnQueryBuilder);
    }

    public void testFromXcontent_WithFilter() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K, TERM_QUERY);
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), knnQueryBuilder.getFilter());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilder.fromXContent(contentParser);
        actualBuilder.equals(knnQueryBuilder);
    }

    @Override
    protected NamedXContentRegistry xContentRegistry() {
        List<NamedXContentRegistry.Entry> list = ClusterModule.getNamedXWriteables();
        SearchPlugin.QuerySpec<?> spec = new SearchPlugin.QuerySpec<>(
            TermQueryBuilder.NAME,
            TermQueryBuilder::new,
            TermQueryBuilder::fromXContent
        );
        list.add(new NamedXContentRegistry.Entry(QueryBuilder.class, spec.getName(), (p, c) -> spec.getParser().fromXContent(p)));
        NamedXContentRegistry registry = new NamedXContentRegistry(list);
        return registry;
    }

    @Override
    protected NamedWriteableRegistry writableRegistry() {
        final List<NamedWriteableRegistry.Entry> entries = ClusterModule.getNamedWriteables();
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, KNNQueryBuilder.NAME, KNNQueryBuilder::new));
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, TermQueryBuilder.NAME, TermQueryBuilder::new));
        return new NamedWriteableRegistry(entries);
    }

    public void testDoToQuery_Normal() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getDimension()).thenReturn(4);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(knnQueryBuilder.getK(), query.getK());
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    public void testDoToQuery_KnnQueryWithFilter() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K, TERM_QUERY);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getDimension()).thenReturn(4);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, ImmutableMap.of());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMethodContext()).thenReturn(knnMethodContext);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query.getClass().isAssignableFrom(KnnFloatVectorQuery.class));
    }

    public void testDoToQuery_FromModel() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);

        // Dimension is -1. In this case, model metadata will need to provide dimension
        when(mockKNNVectorField.getDimension()).thenReturn(-K);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMethodContext()).thenReturn(null);
        String modelId = "test-model-id";
        when(mockKNNVectorField.getModelId()).thenReturn(modelId);

        // Mock the modelDao to return mocked modelMetadata
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getDimension()).thenReturn(4);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        KNNQueryBuilder.initialize(modelDao);

        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(knnQueryBuilder.getK(), query.getK());
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    public void testDoToQuery_InvalidDimensions() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getDimension()).thenReturn(400);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        when(mockKNNVectorField.getDimension()).thenReturn(K);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_InvalidFieldType() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("mynumber", queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        NumberFieldMapper.NumberFieldType mockNumberField = mock(NumberFieldMapper.NumberFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockNumberField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testSerialization() throws Exception {
        assertSerialization(Version.CURRENT, Optional.empty());

        assertSerialization(Version.CURRENT, Optional.of(TERM_QUERY));

        assertSerialization(Version.V_2_3_0, Optional.empty());
    }

    private void assertSerialization(final Version version, final Optional<QueryBuilder> queryBuilderOptional) throws Exception {
        final KNNQueryBuilder knnQueryBuilder = queryBuilderOptional.isPresent()
            ? new KNNQueryBuilder(FIELD_NAME, QUERY_VECTOR, K, queryBuilderOptional.get())
            : new KNNQueryBuilder(FIELD_NAME, QUERY_VECTOR, K);

        final ClusterService clusterService = mockClusterService(version);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            output.setVersion(version);
            output.writeNamedWriteable(knnQueryBuilder);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                in.setVersion(Version.CURRENT);
                final QueryBuilder deserializedQuery = in.readNamedWriteable(QueryBuilder.class);

                assertNotNull(deserializedQuery);
                assertTrue(deserializedQuery instanceof KNNQueryBuilder);
                final KNNQueryBuilder deserializedKnnQueryBuilder = (KNNQueryBuilder) deserializedQuery;
                assertEquals(FIELD_NAME, deserializedKnnQueryBuilder.fieldName());
                assertArrayEquals(QUERY_VECTOR, (float[]) deserializedKnnQueryBuilder.vector(), 0.0f);
                assertEquals(K, deserializedKnnQueryBuilder.getK());
                if (queryBuilderOptional.isPresent()) {
                    assertNotNull(deserializedKnnQueryBuilder.getFilter());
                    assertEquals(queryBuilderOptional.get(), deserializedKnnQueryBuilder.getFilter());
                } else {
                    assertNull(deserializedKnnQueryBuilder.getFilter());
                }
            }
        }
    }
}
