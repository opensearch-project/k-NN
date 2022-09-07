/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.index.Index;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.plugins.SearchPlugin;

import java.io.IOException;
import java.util.List;

import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class KNNQueryBuilderTests extends KNNTestCase {

    public void testInvalidK() {
        float[] queryVector = { 1.0f, 1.0f };

        /**
         * -ve k
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder("myvector", queryVector, -1));

        /**
         * zero k
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder("myvector", queryVector, 0));

        /**
         * k > KNNQueryBuilder.K_MAX
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder("myvector", queryVector, KNNQueryBuilder.K_MAX + 1));
    }

    public void testEmptyVector() {
        /**
         * null query vector
         */
        float[] queryVector = null;
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder("myvector", queryVector, 1));

        /**
         * empty query vector
         */
        float[] queryVector1 = {};
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder("myvector", queryVector1, 1));
    }

    public void testFromXcontent() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("myvector", queryVector, 1);
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
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("myvector", queryVector, 1, QueryBuilders.termQuery("field", "value"));
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

    public void testDoToQuery_Normal() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("myvector", queryVector, 1);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getDimension()).thenReturn(4);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(knnQueryBuilder.getK(), query.getK());
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    public void testDoToQuery_KnnQueryWithFilter() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("myvector", queryVector, 1, QueryBuilders.termQuery("field", "value"));
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getDimension()).thenReturn(4);
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, ImmutableMap.of());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMethodContext()).thenReturn(knnMethodContext);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query instanceof KnnVectorQuery);
    }

    public void testDoToQuery_FromModel() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("myvector", queryVector, 1);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);

        // Dimension is -1. In this case, model metadata will need to provide dimension
        when(mockKNNVectorField.getDimension()).thenReturn(-1);
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
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("myvector", queryVector, 1);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldMapper.KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getDimension()).thenReturn(400);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        when(mockKNNVectorField.getDimension()).thenReturn(1);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_InvalidFieldType() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("mynumber", queryVector, 1);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        NumberFieldMapper.NumberFieldType mockNumberField = mock(NumberFieldMapper.NumberFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockNumberField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }
}
