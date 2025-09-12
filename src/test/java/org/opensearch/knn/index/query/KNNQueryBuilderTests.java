/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.junit.Before;
import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery;
import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static java.util.Collections.emptyMap;
import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;
import static org.opensearch.knn.index.engine.KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH;

public class KNNQueryBuilderTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final int K = 1;
    private static final int EF_SEARCH = 10;
    private static final Map<String, ?> HNSW_METHOD_PARAMS = Map.of("ef_search", EF_SEARCH);
    private static final Float MAX_DISTANCE = 1.0f;
    private static final Float MIN_SCORE = 0.5f;
    private static final TermQueryBuilder TERM_QUERY = QueryBuilders.termQuery("field", "value");
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
    protected static final String TEXT_FIELD_NAME = "some_field";
    protected static final String TEXT_VALUE = "some_value";

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        ClusterSettings clusterSettings = mock(ClusterSettings.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }

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

    public void testInvalidDistance() {
        float[] queryVector = { 1.0f, 1.0f };
        /**
         * null distance
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).maxDistance(null).build()
        );
    }

    public void testInvalidScore() {
        float[] queryVector = { 1.0f, 1.0f };
        /**
         * null min_score
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(null).build()
        );

        /**
         * negative min_score
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(-1.0f).build()
        );

        /**
         * min_score = 0
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(0.0f).build()
        );
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

        /**
         * null query vector with distance
         */
        float[] queryVector2 = null;
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector2).maxDistance(MAX_DISTANCE).build()
        );

        /**
         * empty query vector with distance
         */
        float[] queryVector3 = {};
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector3).maxDistance(MAX_DISTANCE).build()
        );
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
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(knnQueryBuilder.getK(), query.getK());
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    @SneakyThrows
    public void testDoToQuery_whenNormal_whenDoRadiusSearch_whenDistanceThreshold_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        FloatVectorSimilarityQuery query = (FloatVectorSimilarityQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        float resultSimilarity = KNNEngine.LUCENE.distanceToRadialThreshold(MAX_DISTANCE, SpaceType.L2);

        assertTrue(query.toString().contains("resultSimilarity=" + resultSimilarity));
        assertTrue(
            query.toString()
                .contains(
                    "traversalSimilarity="
                        + org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO * resultSimilarity
                )
        );
    }

    @SneakyThrows
    public void testDoToQuery_whenNormal_whenDoRadiusSearch_whenScoreThreshold_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(MIN_SCORE).build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        FloatVectorSimilarityQuery query = (FloatVectorSimilarityQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertTrue(query.toString().contains("resultSimilarity=" + 0.5f));
    }

    @SneakyThrows
    public void testDoToQuery_whenDoRadiusSearch_whenPassNegativeDistance_whenSupportedSpaceType_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        float negativeDistance = -1.0f;

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(negativeDistance)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.INNER_PRODUCT, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);

        assertEquals(negativeDistance, query.getRadius(), 0);
    }

    public void testDoToQuery_whenDoRadiusSearch_whenPassNegativeDistance_whenUnSupportedSpaceType_thenException() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        float negativeDistance = -1.0f;

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(negativeDistance)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    @SneakyThrows
    public void testDoToQuery_whenDoRadiusSearch_whenPassScoreMoreThanOne_whenSupportedSpaceType_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        float score = 5f;

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(score).build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.INNER_PRODUCT, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);

        assertEquals(1 - score, query.getRadius(), 0);
    }

    public void testDoToQuery_whenDoRadiusSearch_whenPassScoreMoreThanOne_whenUnsupportedSpaceType_thenException() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        float score = 5f;
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(score).build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    @SneakyThrows
    public void testDoToQuery_whenPassNegativeDistance_whenSupportedSpaceType_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        float negativeDistance = -1.0f;
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(negativeDistance)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.INNER_PRODUCT, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);

        assertEquals(negativeDistance, query.getRadius(), 0);
    }

    public void testDoToQuery_whenPassNegativeDistance_whenUnSupportedSpaceType_thenException() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        float negativeDistance = -1.0f;

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(negativeDistance)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_whenRadialSearchOnBinaryIndex_thenException() {
        float[] queryVector = { 1.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.HAMMING, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 8));
        Exception e = expectThrows(UnsupportedOperationException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        assertTrue(e.getMessage().contains("Binary data type does not support radial search"));
    }

    public void testDoToQuery_whenRadialSearchOnDiskMode_thenException() {
        float[] queryVector = { 1.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(knnMethodContext);
            }

            @Override
            public int getDimension() {
                return 1;
            }

            public Mode getMode() {
                return Mode.ON_DISK;
            }

            public QuantizationConfig getQuantizationConfig() {
                return QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
            }
        });
        Exception e = expectThrows(UnsupportedOperationException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        assertEquals("Radial search is not supported for indices which have quantization enabled", e.getMessage());
    }

    public void testDoToQuery_KnnQueryWithFilter_Lucene() throws Exception {
        // Given
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(TERM_QUERY)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        // When
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);

        // Then
        assertNotNull(query);
        assertTrue(query.getClass().isAssignableFrom(LuceneEngineKnnVectorQuery.class));
    }

    @SneakyThrows
    public void testDoToQuery_whenDoRadiusSearch_whenDistanceThreshold_whenFilter_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .filter(TERM_QUERY)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query.getClass().isAssignableFrom(FloatVectorSimilarityQuery.class));
    }

    @SneakyThrows
    public void testDoToQuery_whenDoRadiusSearch_whenScoreThreshold_whenFilter_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .filter(TERM_QUERY)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query.getClass().isAssignableFrom(FloatVectorSimilarityQuery.class));
    }

    @SneakyThrows
    public void testDoToQuery_WhenknnQueryWithFilterAndFaissEngine_thenSuccess() {
        // Given
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        // When
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(TERM_QUERY)
            .methodParameters(HNSW_METHOD_PARAMS)
            .build();

        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);

        // Then
        assertNotNull(query);
        assertTrue(query instanceof KNNQuery);
        assertEquals(HNSW_METHOD_PARAMS, ((KNNQuery) query).getMethodParameters());
    }

    public void testDoToQuery_ThrowsIllegalArgumentExceptionForUnknownMethodParameter() {

        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        when(mockQueryShardContext.index()).thenReturn(new Index("dummy", "dummy"));
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            new MethodComponentContext("hnsw", Map.of())
        );
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .methodParameters(Map.of("nprobes", 10))
            .build();

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_whenknnQueryWithFilterAndNmsLibEngine_thenException() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K, TERM_QUERY);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.NMSLIB, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_whenMemoryOptimizedSearchIsEnabled() {
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, true, VectorDataType.FLOAT, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, true, VectorDataType.FLOAT, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, true, VectorDataType.BYTE, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, true, VectorDataType.BYTE, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, false, VectorDataType.FLOAT, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, false, VectorDataType.FLOAT, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, false, VectorDataType.BYTE, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, false, VectorDataType.BYTE, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, true, VectorDataType.BINARY, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, true, VectorDataType.BINARY, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, false, VectorDataType.BINARY, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(true, false, VectorDataType.BINARY, false);

        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, true, VectorDataType.FLOAT, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, true, VectorDataType.FLOAT, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, true, VectorDataType.BYTE, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, true, VectorDataType.BYTE, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, false, VectorDataType.FLOAT, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, false, VectorDataType.FLOAT, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, false, VectorDataType.BYTE, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, false, VectorDataType.BYTE, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, false, VectorDataType.BINARY, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, false, VectorDataType.BINARY, false);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, true, VectorDataType.BINARY, true);
        do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(false, true, VectorDataType.BINARY, false);
    }

    private void do_testDoToQuery_whenMemoryOptimizedSearchIsEnabled(
        final boolean memoryOptimizedSearchEnabled,
        final boolean memoryOptimizedSearchEnabledInField,
        VectorDataType vectorDataType,
        final boolean doRescore
    ) {

        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.isMemoryOptimizedKnnSearchModeEnabled(any()))
                .thenReturn(memoryOptimizedSearchEnabled);

            // Query vector
            final float[] queryVector;
            final int dimension = 8;
            if (vectorDataType == VectorDataType.BINARY) {
                // Binary vector, e.g. 77 = 0b1001101, [0, 1, 0, 0, 1, 1, 0, 1]
                queryVector = new float[] { 77f };
            } else {
                queryVector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            }
            KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K, TERM_QUERY);

            // Query shard context
            Index dummyIndex = new Index("dummy", "dummy");
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            when(mockQueryShardContext.index()).thenReturn(dummyIndex);

            // Field type
            KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
            when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
            when(mockKNNVectorField.isMemoryOptimizedSearchAvailable()).thenReturn(memoryOptimizedSearchEnabledInField);
            when(mockKNNVectorField.getVectorDataType()).thenReturn(vectorDataType);
            if (doRescore) {
                when(mockKNNVectorField.resolveRescoreContext(any())).thenReturn(mock(RescoreContext.class));
            }

            // Method context
            MethodComponentContext methodComponentContext = new MethodComponentContext(
                org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
                ImmutableMap.of()
            );
            final KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);

            // KNN mapping config
            when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, dimension));

            // Execute `doToQuery`
            final Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
            // If memory optimized search is on then, use Lucene query
            final KNNQuery knnQuery;
            if (doRescore) {
                assertTrue(query instanceof NativeEngineKnnVectorQuery);
                knnQuery = ((NativeEngineKnnVectorQuery) query).getKnnQuery();
            } else {
                assertFalse(query instanceof NativeEngineKnnVectorQuery);
                knnQuery = (KNNQuery) query;
            }

            final boolean memoryOptimizedEnabled = memoryOptimizedSearchEnabled && memoryOptimizedSearchEnabledInField;
            if (memoryOptimizedEnabled) {
                if (vectorDataType == VectorDataType.FLOAT) {
                    assertEquals(queryVector.length, knnQuery.getQueryVector().length);
                } else if (vectorDataType == VectorDataType.BYTE) {
                    assertEquals(queryVector.length, knnQuery.getByteQueryVector().length);
                } else if (vectorDataType == VectorDataType.BINARY) {
                    assertEquals(queryVector.length, knnQuery.getByteQueryVector().length);
                }
            }
        }
    }

    @SneakyThrows
    public void testDoToQuery_FromModel() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);

        // Dimension is -1. In this case, model metadata will need to provide dimension
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        String modelId = "test-model-id";
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForModelMapping(modelId, 4));

        // Mock the modelDao to return mocked modelMetadata
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(modelMetadata.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
        when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
        when(modelMetadata.getMethodComponentContext()).thenReturn(new MethodComponentContext("ivf", emptyMap()));
        when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.DEFAULT);
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        KNNQueryBuilder.initialize(modelDao);

        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(knnQueryBuilder.getK(), query.getK());
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    @SneakyThrows
    public void testDoToQuery_whenFromModel_whenDoRadiusSearch_whenDistanceThreshold_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);

        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        String modelId = "test-model-id";
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForModelMapping(modelId, 4));

        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(modelMetadata.getSpaceType()).thenReturn(SpaceType.L2);
        when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
        when(modelMetadata.getMethodComponentContext()).thenReturn(new MethodComponentContext("ivf", emptyMap()));
        when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.DEFAULT);
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        KNNQueryBuilder.initialize(modelDao);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(knnQueryBuilder.getMaxDistance(), query.getRadius(), 0);
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    @SneakyThrows
    public void testDoToQuery_whenFromModel_whenDoRadiusSearch_whenScoreThreshold_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(MIN_SCORE).build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        String modelId = "test-model-id";
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForModelMapping(modelId, 4));

        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(modelMetadata.getSpaceType()).thenReturn(SpaceType.L2);
        when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
        when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.DEFAULT);
        when(modelMetadata.getMethodComponentContext()).thenReturn(new MethodComponentContext("ivf", emptyMap()));
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        KNNQueryBuilder.initialize(modelDao);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);

        assertEquals(1 / knnQueryBuilder.getMinScore() - 1, query.getRadius(), 0);
        assertEquals(knnQueryBuilder.fieldName(), query.getField());
        assertEquals(knnQueryBuilder.vector(), query.getQueryVector());
    }

    public void testDoToQuery_InvalidDimensions() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 400));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), K));
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

    public void testDoToQuery_InvalidZeroFloatVector() {
        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> knnQueryBuilder.doToQuery(mockQueryShardContext)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception.getMessage()
        );
    }

    public void testDoToQuery_InvalidZeroByteVector() {
        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BYTE);
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> knnQueryBuilder.doToQuery(mockQueryShardContext)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception.getMessage()
        );
    }

    public void testSerialization() throws Exception {
        // For k-NN search
        assertSerialization(Version.CURRENT, Optional.empty(), K, null, null, null, null);
        assertSerialization(Version.CURRENT, Optional.empty(), K, Map.of("ef_search", EF_SEARCH), null, null, null);
        assertSerialization(Version.CURRENT, Optional.of(TERM_QUERY), K, Map.of("ef_search", EF_SEARCH), null, null, null);
        assertSerialization(Version.V_2_3_0, Optional.empty(), K, Map.of("ef_search", EF_SEARCH), null, null, null);
        assertSerialization(Version.V_2_3_0, Optional.empty(), K, null, null, null, null);

        // For distance threshold search
        assertSerialization(Version.CURRENT, Optional.empty(), null, null, null, MAX_DISTANCE, null);
        assertSerialization(Version.CURRENT, Optional.of(TERM_QUERY), null, null, null, MAX_DISTANCE, null);

        // For score threshold search
        assertSerialization(Version.CURRENT, Optional.empty(), null, null, null, MIN_SCORE, null);
        assertSerialization(Version.CURRENT, Optional.of(TERM_QUERY), null, null, null, MIN_SCORE, null);

        // Test rescore
        assertSerialization(Version.V_2_3_0, Optional.empty(), K, null, null, null, RescoreContext.getDefault());
        assertSerialization(Version.CURRENT, Optional.empty(), K, null, null, null, RescoreContext.getDefault());
    }

    private void assertSerialization(
        final Version version,
        final Optional<QueryBuilder> queryBuilderOptional,
        Integer k,
        Map<String, ?> methodParameters,
        Float distance,
        Float score,
        RescoreContext rescoreContext
    ) throws Exception {
        final KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .maxDistance(distance)
            .minScore(score)
            .k(k)
            .methodParameters(methodParameters)
            .filter(queryBuilderOptional.orElse(null))
            .rescoreContext(rescoreContext)
            .build();

        final ClusterService clusterService = mockClusterService(version);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            output.setVersion(version);
            output.writeNamedWriteable(knnQueryBuilder);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                in.setVersion(version);
                final QueryBuilder deserializedQuery = in.readNamedWriteable(QueryBuilder.class);

                assertNotNull(deserializedQuery);
                assertTrue(deserializedQuery instanceof KNNQueryBuilder);
                final KNNQueryBuilder deserializedKnnQueryBuilder = (KNNQueryBuilder) deserializedQuery;
                assertEquals(FIELD_NAME, deserializedKnnQueryBuilder.fieldName());
                assertArrayEquals(QUERY_VECTOR, (float[]) deserializedKnnQueryBuilder.vector(), 0.0f);
                if (k != null) {
                    assertEquals(k.intValue(), deserializedKnnQueryBuilder.getK());
                } else if (distance != null) {
                    assertEquals(distance.floatValue(), deserializedKnnQueryBuilder.getMaxDistance(), 0.0f);
                } else {
                    assertEquals(score.floatValue(), deserializedKnnQueryBuilder.getMinScore(), 0.0f);
                }
                if (queryBuilderOptional.isPresent()) {
                    assertNotNull(deserializedKnnQueryBuilder.getFilter());
                    assertEquals(queryBuilderOptional.get(), deserializedKnnQueryBuilder.getFilter());
                } else {
                    assertNull(deserializedKnnQueryBuilder.getFilter());
                }
                assertMethodParameters(version, methodParameters, deserializedKnnQueryBuilder.getMethodParameters());
                assertRescore(version, rescoreContext, deserializedKnnQueryBuilder.getRescoreContext());
            }
        }
    }

    private void assertMethodParameters(Version version, Map<String, ?> expectedMethodParameters, Map<String, ?> actualMethodParameters) {
        if (!version.onOrAfter(Version.V_2_16_0)) {
            assertNull(actualMethodParameters);
        } else if (expectedMethodParameters != null) {
            if (version.onOrAfter(Version.V_2_16_0)) {
                assertEquals(expectedMethodParameters.get("ef_search"), actualMethodParameters.get("ef_search"));
            }
        }
    }

    private void assertRescore(Version version, RescoreContext expectedRescoreContext, RescoreContext actualRescoreContext) {
        if (!version.onOrAfter(Version.V_2_17_0)) {
            assertNull(actualRescoreContext);
            return;
        }

        if (expectedRescoreContext != null) {
            assertNotNull(actualRescoreContext);
            assertEquals(expectedRescoreContext.getOversampleFactor(), actualRescoreContext.getOversampleFactor(), 0.0f);
        }
    }

    public void testIgnoreUnmapped() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder.Builder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .ignoreUnmapped(true);
        assertTrue(knnQueryBuilder.build().isIgnoreUnmapped());
        Query query = knnQueryBuilder.build().doToQuery(mock(QueryShardContext.class));
        assertNotNull(query);
        assertThat(query, instanceOf(MatchNoDocsQuery.class));
        knnQueryBuilder.ignoreUnmapped(false);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.build().doToQuery(mock(QueryShardContext.class)));
    }

    public void testRadialSearch_whenUnsupportedEngine_thenThrowException() {
        List<KNNEngine> unsupportedEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !ENGINES_SUPPORTING_RADIAL_SEARCH.contains(knnEngine))
            .filter(knnEngine -> knnEngine != KNNEngine.UNDEFINED)
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : unsupportedEngines) {
            KNNMethodContext knnMethodContext = new KNNMethodContext(
                knnEngine,
                SpaceType.L2,
                new MethodComponentContext(org.opensearch.knn.common.KNNConstants.METHOD_HNSW, ImmutableMap.of())
            );

            KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(QUERY_VECTOR)
                .maxDistance(MAX_DISTANCE)
                .build();

            KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            Index dummyIndex = new Index("dummy", "dummy");
            when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
            when(mockQueryShardContext.index()).thenReturn(dummyIndex);
            when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
            when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

            expectThrows(UnsupportedOperationException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        }
    }

    public void testRadialSearch_whenEfSearchIsSet_whenLuceneEngine_thenThrowException() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(org.opensearch.knn.common.KNNConstants.METHOD_HNSW, ImmutableMap.of())
        );

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .maxDistance(MAX_DISTANCE)
            .methodParameters(Map.of("ef_search", EF_SEARCH))
            .build();

        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        Index dummyIndex = new Index("dummy", "dummy");
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    @SneakyThrows
    public void testRadialSearch_whenEfSearchIsSet_whenFaissEngine_thenSuccess() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(org.opensearch.knn.common.KNNConstants.METHOD_HNSW, ImmutableMap.of())
        );

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .minScore(MIN_SCORE)
            .methodParameters(Map.of("ef_search", EF_SEARCH))
            .build();

        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        Index dummyIndex = new Index("dummy", "dummy");
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(1000);

        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertEquals(1 / MIN_SCORE - 1, query.getRadius(), 0);
    }

    public void testDoToQuery_whenBinary_thenValid() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        byte[] expectedQueryVector = { 1, 2, 3, 4 };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultBinaryKNNMethodContext(), 32));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNQuery query = (KNNQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertArrayEquals(expectedQueryVector, query.getByteQueryVector());
        assertNull(query.getQueryVector());
    }

    public void testDoToQuery_whenBinaryWithInvalidDimension_thenException() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultBinaryKNNMethodContext(), 8));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Exception ex = expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        assertTrue(ex.getMessage(), ex.getMessage().contains("invalid dimension"));
    }

    @SneakyThrows
    public void testDoRewrite_whenNoFilter_thenSuccessful() {
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, QUERY_VECTOR, K);
        QueryBuilder rewritten = knnQueryBuilder.rewrite(mock(QueryRewriteContext.class));
        assertEquals(knnQueryBuilder, rewritten);
    }

    @SneakyThrows
    public void testDoRewrite_whenFilterSet_thenSuccessful() {
        // Given
        QueryBuilder filter = mock(QueryBuilder.class);
        QueryBuilder rewrittenFilter = mock(QueryBuilder.class);
        QueryRewriteContext context = mock(QueryRewriteContext.class);
        when(filter.rewrite(context)).thenReturn(rewrittenFilter);
        KNNQueryBuilder expected = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .filter(rewrittenFilter)
            .k(K)
            .build();

        // When
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).filter(filter).k(K).build();

        QueryBuilder actual = knnQueryBuilder.rewrite(context);

        assertEquals(knnQueryBuilder, KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).filter(filter).k(K).build());

        // Then
        assertEquals(expected, actual);
    }

    @SneakyThrows
    public void testFilter() {
        // Test for Null Case
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, QUERY_VECTOR, K);
        KNNQueryBuilder updatedKnnQueryBuilder = (KNNQueryBuilder) knnQueryBuilder.filter(null);
        assertEquals(knnQueryBuilder, updatedKnnQueryBuilder);

        // Test for valid case
        /*
                return KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(vector)
            .k(k)
            .maxDistance(maxDistance)
            .minScore(minScore)
            .methodParameters(methodParameters)
            .filter(filterToBeAdded)
            .ignoreUnmapped(ignoreUnmapped)
            .rescoreContext(rescoreContext)
            .expandNested(expandNested)
            .build();
         */
        knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).filter(TERM_QUERY).k(K).build();
        updatedKnnQueryBuilder = (KNNQueryBuilder) knnQueryBuilder.filter(TERM_QUERY);
        BoolQueryBuilder expectedUpdatedQueryFilter = new BoolQueryBuilder();
        expectedUpdatedQueryFilter.must(TERM_QUERY);
        expectedUpdatedQueryFilter.filter(TERM_QUERY);
        assertEquals(knnQueryBuilder.fieldName(), updatedKnnQueryBuilder.fieldName());
        assertEquals(knnQueryBuilder.vector(), updatedKnnQueryBuilder.vector());
        assertEquals(knnQueryBuilder.getK(), updatedKnnQueryBuilder.getK());
        assertEquals(knnQueryBuilder.getMaxDistance(), updatedKnnQueryBuilder.getMaxDistance());
        assertEquals(knnQueryBuilder.getMinScore(), updatedKnnQueryBuilder.getMinScore());
        assertEquals(knnQueryBuilder.getMethodParameters(), updatedKnnQueryBuilder.getMethodParameters());
        assertEquals(knnQueryBuilder.isIgnoreUnmapped(), updatedKnnQueryBuilder.isIgnoreUnmapped());
        assertEquals(knnQueryBuilder.getRescoreContext(), updatedKnnQueryBuilder.getRescoreContext());
        assertEquals(knnQueryBuilder.getExpandNested(), updatedKnnQueryBuilder.getExpandNested());
        assertEquals(expectedUpdatedQueryFilter, updatedKnnQueryBuilder.getFilter());

        // Test for queryBuilder without filter initialized where filter function would
        // simply assign filter to its filter field.
        knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(K).build();
        updatedKnnQueryBuilder = (KNNQueryBuilder) knnQueryBuilder.filter(TERM_QUERY);
        assertEquals(knnQueryBuilder.fieldName(), updatedKnnQueryBuilder.fieldName());
        assertEquals(knnQueryBuilder.vector(), updatedKnnQueryBuilder.vector());
        assertEquals(knnQueryBuilder.getK(), updatedKnnQueryBuilder.getK());
        assertEquals(knnQueryBuilder.getMaxDistance(), updatedKnnQueryBuilder.getMaxDistance());
        assertEquals(knnQueryBuilder.getMinScore(), updatedKnnQueryBuilder.getMinScore());
        assertEquals(knnQueryBuilder.getMethodParameters(), updatedKnnQueryBuilder.getMethodParameters());
        assertEquals(knnQueryBuilder.isIgnoreUnmapped(), updatedKnnQueryBuilder.isIgnoreUnmapped());
        assertEquals(knnQueryBuilder.getRescoreContext(), updatedKnnQueryBuilder.getRescoreContext());
        assertEquals(knnQueryBuilder.getExpandNested(), updatedKnnQueryBuilder.getExpandNested());
        assertEquals(TERM_QUERY, updatedKnnQueryBuilder.getFilter());
    }
}
