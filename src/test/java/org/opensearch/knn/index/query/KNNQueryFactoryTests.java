/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockedConstruction;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.index.search.NestedHelper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.lucenelib.ExpandNestedDocsQuery;
import org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery;
import org.opensearch.knn.index.query.lucenelib.OSDiversifyingChildrenByteKnnVectorQuery;
import org.opensearch.knn.index.query.lucenelib.OSDiversifyingChildrenFloatKnnVectorQuery;
import org.opensearch.knn.index.query.lucenelib.OSKnnByteVectorQuery;
import org.opensearch.knn.index.query.lucenelib.OSKnnFloatVectorQuery;
import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

public class KNNQueryFactoryTests extends KNNTestCase {
    private static final String FILTER_FILED_NAME = "foo";
    private static final String FILTER_FILED_VALUE = "fooval";
    private static final QueryBuilder FILTER_QUERY_BUILDER = new TermQueryBuilder(FILTER_FILED_NAME, FILTER_FILED_VALUE);
    private static final Query FILTER_QUERY = new TermQuery(new Term(FILTER_FILED_NAME, FILTER_FILED_VALUE));
    private final int testQueryDimension = 17;
    private final float[] testQueryVector = new float[testQueryDimension];
    private final byte[] testByteQueryVector = new byte[testQueryDimension];
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int testK = 10;
    private final Map<String, ?> methodParameters = Map.of(METHOD_PARAMETER_EF_SEARCH, 100);

    @Mock
    ClusterSettings clusterSettings;
    @Mock
    ClusterState clusterState;
    @Mock
    Metadata metadata;
    @Mock
    IndexMetadata indexMetadata;

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.state()).thenReturn(clusterState);
        when(clusterState.getMetadata()).thenReturn(metadata);
        when(metadata.index(testIndexName)).thenReturn(indexMetadata);
        when(indexMetadata.getSettings()).thenReturn(Settings.EMPTY);
        when(indexMetadata.getCreationVersion()).thenReturn(Version.CURRENT);
        KNNSettings.state().setClusterService(clusterService);
    }

    public void testCreateCustomKNNQuery() {
        for (KNNEngine knnEngine : KNNEngine.getEnginesThatCreateCustomSegmentFiles()) {
            Query query = KNNQueryFactory.create(
                BaseQueryFactory.CreateQueryRequest.builder()
                    .knnEngine(knnEngine)
                    .indexName(testIndexName)
                    .fieldName(testFieldName)
                    .vector(testQueryVector)
                    .k(testK)
                    .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                    .build()
            );
            assertTrue(query instanceof KNNQuery);
            assertEquals(testIndexName, ((KNNQuery) query).getIndexName());
            assertEquals(testFieldName, ((KNNQuery) query).getField());
            assertEquals(testQueryVector, ((KNNQuery) query).getQueryVector());
            assertEquals(testK, ((KNNQuery) query).getK());

            query = KNNQueryFactory.create(
                BaseQueryFactory.CreateQueryRequest.builder()
                    .knnEngine(knnEngine)
                    .indexName(testIndexName)
                    .fieldName(testFieldName)
                    .vector(testQueryVector)
                    .k(testK)
                    .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                    .build()
            );

            assertTrue(query instanceof KNNQuery);
            assertEquals(testIndexName, ((KNNQuery) query).getIndexName());
            assertEquals(testFieldName, ((KNNQuery) query).getField());
            assertEquals(testQueryVector, ((KNNQuery) query).getQueryVector());
            assertEquals(testK, ((KNNQuery) query).getK());
        }
    }

    public void testCreateLuceneDefaultQuery() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            Query query = KNNQueryFactory.create(
                BaseQueryFactory.CreateQueryRequest.builder()
                    .knnEngine(knnEngine)
                    .indexName(testIndexName)
                    .fieldName(testFieldName)
                    .vector(testQueryVector)
                    .k(testK)
                    .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                    .build()
            );
            assertEquals(LuceneEngineKnnVectorQuery.class, query.getClass());
        }
    }

    public void testLuceneFloatVectorQuery() {
        Query actualQuery1 = buildLuceneFloatQuery(methodParameters);

        // efsearch > k and efSearch was set using methodParameters
        int luceneK = getLuceneK(testK, methodParameters);
        Query expectedQuery1 = new LuceneEngineKnnVectorQuery(
            new OSKnnFloatVectorQuery(testFieldName, testQueryVector, luceneK, null, testK)
        );
        assertEquals(expectedQuery1, actualQuery1);
        assertEquals(luceneK, methodParameters.get(METHOD_PARAMETER_EF_SEARCH));

        // efsearch < k and efSearch was set using methodParameters
        Map<String, ?> methodParams = Map.of(METHOD_PARAMETER_EF_SEARCH, 1);
        actualQuery1 = buildLuceneFloatQuery(methodParams);
        luceneK = getLuceneK(testK, methodParams);
        expectedQuery1 = new LuceneEngineKnnVectorQuery(new OSKnnFloatVectorQuery(testFieldName, testQueryVector, luceneK, null, testK));
        assertEquals(expectedQuery1, actualQuery1);
        assertEquals(luceneK, testK);

        // efsearch > k and efSearch was set using index setting
        int efSearchIndexSettingValue = 110;
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any())).thenReturn(efSearchIndexSettingValue);
            luceneK = getLuceneK(testK, null);
            actualQuery1 = buildLuceneFloatQuery(null);
            expectedQuery1 = new LuceneEngineKnnVectorQuery(
                new OSKnnFloatVectorQuery(testFieldName, testQueryVector, luceneK, null, testK)
            );
            assertEquals(expectedQuery1, actualQuery1);
            assertEquals(luceneK, efSearchIndexSettingValue);
        }

        // efsearch < k and efSearch was set using index setting
        efSearchIndexSettingValue = 2;
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any())).thenReturn(efSearchIndexSettingValue);
            luceneK = getLuceneK(testK, null);
            actualQuery1 = buildLuceneFloatQuery(null);
            expectedQuery1 = new LuceneEngineKnnVectorQuery(
                new OSKnnFloatVectorQuery(testFieldName, testQueryVector, luceneK, null, testK)
            );
            assertEquals(expectedQuery1, actualQuery1);
            assertEquals(luceneK, testK);
        }

        // efSearch was not set using methodParameters or index setting then the default value of efSearch will be used
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any()))
                .thenReturn(IndexHyperParametersUtil.getHNSWEFSearchValue(Version.CURRENT));
            luceneK = getLuceneK(testK, null);
            actualQuery1 = buildLuceneFloatQuery(null);
            expectedQuery1 = new LuceneEngineKnnVectorQuery(
                new OSKnnFloatVectorQuery(testFieldName, testQueryVector, luceneK, null, testK)
            );
            assertEquals(expectedQuery1, actualQuery1);
            assertEquals(luceneK, IndexHyperParametersUtil.getHNSWEFSearchValue(Version.CURRENT));
        }
    }

    public void testLuceneByteVectorQuery() {
        Query actualQuery1 = buildLuceneByteQuery(methodParameters);

        // efsearch > k and efSearch was set using methodParameters
        int luceneK = getLuceneK(testK, methodParameters);
        Query expectedQuery1 = new LuceneEngineKnnVectorQuery(
            new OSKnnByteVectorQuery(testFieldName, testByteQueryVector, luceneK, null, testK)
        );
        assertEquals(expectedQuery1, actualQuery1);
        assertEquals(luceneK, methodParameters.get(METHOD_PARAMETER_EF_SEARCH));

        // efsearch < k and efSearch was set using methodParameters
        Map<String, ?> methodParams = Map.of(METHOD_PARAMETER_EF_SEARCH, 1);
        actualQuery1 = buildLuceneByteQuery(methodParams);
        luceneK = getLuceneK(testK, methodParams);
        expectedQuery1 = new LuceneEngineKnnVectorQuery(new OSKnnByteVectorQuery(testFieldName, testByteQueryVector, luceneK, null, testK));
        assertEquals(expectedQuery1, actualQuery1);
        assertEquals(luceneK, testK);

        // efsearch > k and efSearch was set using index setting
        int efSearchIndexSettingValue = 110;
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any())).thenReturn(efSearchIndexSettingValue);
            luceneK = getLuceneK(testK, null);
            actualQuery1 = buildLuceneByteQuery(null);
            expectedQuery1 = new LuceneEngineKnnVectorQuery(
                new OSKnnByteVectorQuery(testFieldName, testByteQueryVector, luceneK, null, testK)
            );
            assertEquals(expectedQuery1, actualQuery1);
            assertEquals(luceneK, efSearchIndexSettingValue);
        }

        // efsearch < k and efSearch was set using index setting
        efSearchIndexSettingValue = 2;
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any())).thenReturn(efSearchIndexSettingValue);
            luceneK = getLuceneK(testK, null);
            actualQuery1 = buildLuceneByteQuery(null);
            expectedQuery1 = new LuceneEngineKnnVectorQuery(
                new OSKnnByteVectorQuery(testFieldName, testByteQueryVector, luceneK, null, testK)
            );
            assertEquals(expectedQuery1, actualQuery1);
            assertEquals(luceneK, testK);
        }

        // efSearch was not set using methodParameters or index setting then the default value of efSearch will be used
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            // Index setting mocking
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any()))
                .thenReturn(IndexHyperParametersUtil.getHNSWEFSearchValue(Version.CURRENT));
            luceneK = getLuceneK(testK, null);
            actualQuery1 = buildLuceneByteQuery(null);
            expectedQuery1 = new LuceneEngineKnnVectorQuery(
                new OSKnnByteVectorQuery(testFieldName, testByteQueryVector, luceneK, null, testK)
            );
            assertEquals(expectedQuery1, actualQuery1);
            assertEquals(luceneK, IndexHyperParametersUtil.getHNSWEFSearchValue(Version.CURRENT));
        }
    }

    public void testCreateLuceneQueryWithFilter() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            MappedFieldType testMapper = mock(MappedFieldType.class);
            when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
            final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .k(testK)
                .context(mockQueryShardContext)
                .filter(FILTER_QUERY_BUILDER)
                .build();
            Query query = KNNQueryFactory.create(createQueryRequest);
            assertEquals(LuceneEngineKnnVectorQuery.class, query.getClass());
        }
    }

    public void testCreateFaissQueryWithFilter_withValidValues_thenSuccess() {
        // Given
        final KNNEngine knnEngine = KNNEngine.FAISS;
        final QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(testMapper.termQuery(Mockito.any(), Mockito.eq(mockQueryShardContext))).thenReturn(FILTER_QUERY);

        final KNNQuery expectedQuery = KNNQuery.builder()
            .indexName(testIndexName)
            .filterQuery(FILTER_QUERY)
            .field(testFieldName)
            .queryVector(testQueryVector)
            .k(testK)
            .methodParameters(methodParameters)
            .vectorDataType(VectorDataType.FLOAT)
            .build();

        // When
        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .k(testK)
            .methodParameters(methodParameters)
            .vectorDataType(VectorDataType.FLOAT)
            .context(mockQueryShardContext)
            .filter(FILTER_QUERY_BUILDER)
            .build();

        final Query actual = KNNQueryFactory.create(createQueryRequest);

        // Then
        assertEquals(expectedQuery, actual);
    }

    public void testCreateFaissQueryWithFilter_withValidValues_nullEfSearch_thenSuccess() {
        // Given
        final KNNEngine knnEngine = KNNEngine.FAISS;
        final QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(testMapper.termQuery(Mockito.any(), Mockito.eq(mockQueryShardContext))).thenReturn(FILTER_QUERY);

        final KNNQuery expectedQuery = KNNQuery.builder()
            .indexName(testIndexName)
            .filterQuery(FILTER_QUERY)
            .field(testFieldName)
            .queryVector(testQueryVector)
            .k(testK)
            .build();

        // When
        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .k(testK)
            .vectorDataType(VectorDataType.FLOAT)
            .context(mockQueryShardContext)
            .filter(FILTER_QUERY_BUILDER)
            .build();

        final Query actual = KNNQueryFactory.create(createQueryRequest);

        // Then
        assertEquals(expectedQuery, actual);
    }

    public void testCreate_whenLuceneWithParentFilter_thenReturnDiversifyingQuery() {
        validateDiversifyingQueryWithParentFilter(VectorDataType.BYTE, LuceneEngineKnnVectorQuery.class);
        validateDiversifyingQueryWithParentFilter(VectorDataType.FLOAT, LuceneEngineKnnVectorQuery.class);
    }

    public void testCreate_whenNestedVectorFiledAndNonNestedFilterField_thenReturnToChildBlockJoinQueryForFilters() {
        MapperService mockMapperService = mock(MapperService.class);
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        when(mockQueryShardContext.getMapperService()).thenReturn(mockMapperService);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(testMapper.termQuery(Mockito.any(), Mockito.eq(mockQueryShardContext))).thenReturn(FILTER_QUERY);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);
        MockedConstruction<NestedHelper> mockedNestedHelper = Mockito.mockConstruction(
            NestedHelper.class,
            (mock, context) -> when(mock.mightMatchNestedDocs(FILTER_QUERY)).thenReturn(false)
        );

        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .k(testK)
            .vectorDataType(VectorDataType.FLOAT)
            .context(mockQueryShardContext)
            .filter(FILTER_QUERY_BUILDER)
            .build();
        KNNQuery query = (KNNQuery) KNNQueryFactory.create(createQueryRequest);
        mockedNestedHelper.close();
        assertEquals(ToChildBlockJoinQuery.class, query.getFilterQuery().getClass());
    }

    public void testCreate_whenNestedVectorAndFilterField_thenReturnToChildBlockJoinQueryForFilters() {
        MapperService mockMapperService = mock(MapperService.class);
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        when(mockQueryShardContext.getMapperService()).thenReturn(mockMapperService);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(testMapper.termQuery(Mockito.any(), Mockito.eq(mockQueryShardContext))).thenReturn(FILTER_QUERY);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);
        MockedConstruction<NestedHelper> mockedNestedHelper = Mockito.mockConstruction(
            NestedHelper.class,
            (mock, context) -> when(mock.mightMatchNestedDocs(FILTER_QUERY)).thenReturn(true)
        );

        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .k(testK)
            .vectorDataType(VectorDataType.FLOAT)
            .context(mockQueryShardContext)
            .filter(FILTER_QUERY_BUILDER)
            .build();
        KNNQuery query = (KNNQuery) KNNQueryFactory.create(createQueryRequest);
        mockedNestedHelper.close();
        assertEquals(ToChildBlockJoinQuery.class, query.getFilterQuery().getClass());
    }

    public void testCreate_whenFaissWithParentFilter_thenSuccess() {
        final KNNEngine knnEngine = KNNEngine.FAISS;
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);
        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .k(testK)
            .vectorDataType(VectorDataType.FLOAT)
            .context(mockQueryShardContext)
            .build();
        final Query query = KNNQueryFactory.create(createQueryRequest);
        assertTrue(query instanceof KNNQuery);
        assertEquals(testIndexName, ((KNNQuery) query).getIndexName());
        assertEquals(testFieldName, ((KNNQuery) query).getField());
        assertEquals(testQueryVector, ((KNNQuery) query).getQueryVector());
        assertEquals(testK, ((KNNQuery) query).getK());
        assertEquals(parentFilter, ((KNNQuery) query).getParentsFilter());
    }

    private void validateDiversifyingQueryWithParentFilter(final VectorDataType type, final Class expectedQueryClass) {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            MappedFieldType testMapper = mock(MappedFieldType.class);
            when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
            BitSetProducer parentFilter = mock(BitSetProducer.class);
            when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);
            final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .byteVector(testByteQueryVector)
                .vectorDataType(type)
                .k(testK)
                .context(mockQueryShardContext)
                .filter(FILTER_QUERY_BUILDER)
                .build();
            Query query = KNNQueryFactory.create(createQueryRequest);
            assertEquals(expectedQueryClass, query.getClass());
        }
    }

    public void testCreate_whenBinary_thenSuccess() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);

        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .byteVector(testByteQueryVector)
            .vectorDataType(VectorDataType.BINARY)
            .k(testK)
            .context(mockQueryShardContext)
            .filter(FILTER_QUERY_BUILDER)
            .build();
        Query query = KNNQueryFactory.create(createQueryRequest);
        assertTrue(query instanceof KNNQuery);
        assertNotNull(((KNNQuery) query).getByteQueryVector());
        assertNull(((KNNQuery) query).getQueryVector());
    }

    public void testCreate_whenRescoreContext() {
        test_rescoreContext_faiss(RescoreContext.getDefault());
        test_rescoreContext_faiss(null);
    }

    private void test_rescoreContext_faiss(RescoreContext rescoreContext) {
        // Given
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);

        final KNNQuery expected = KNNQuery.builder()
            .field(testFieldName)
            .indexName(testIndexName)
            .byteQueryVector(testByteQueryVector)
            .k(testK)
            .parentsFilter(parentFilter)
            .vectorDataType(VectorDataType.BINARY)
            .rescoreContext(rescoreContext)
            .build();

        // When
        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .byteVector(testByteQueryVector)
            .vectorDataType(VectorDataType.BINARY)
            .k(testK)
            .context(mockQueryShardContext)
            .rescoreContext(rescoreContext)
            .build();
        Query query = KNNQueryFactory.create(createQueryRequest);

        // Then
        if (rescoreContext != null) {
            assertEquals(expected, ((NativeEngineKnnVectorQuery) query).getKnnQuery());
        } else {
            assertEquals(expected, query);
        }
    }

    public void testCreate_whenExpandNestedDocsQueryWithFaiss_thenCreateNativeEngineKNNVectorQuery() {
        testExpandNestedDocsQuery(KNNEngine.FAISS, NativeEngineKnnVectorQuery.class, VectorDataType.values()[randomInt(2)], true);
        testExpandNestedDocsQuery(KNNEngine.FAISS, KNNQuery.class, VectorDataType.values()[randomInt(2)], false);
    }

    public void testCreate_whenExpandNestedDocsQueryWithNmslib_thenCreateKNNQuery() {
        testExpandNestedDocsQuery(KNNEngine.NMSLIB, KNNQuery.class, VectorDataType.FLOAT, true);
        testExpandNestedDocsQuery(KNNEngine.NMSLIB, KNNQuery.class, VectorDataType.FLOAT, false);
    }

    public void testCreate_whenExpandNestedDocsQueryWithLucene_thenCreateExpandNestedDocsQuery() {
        testExpandNestedDocsQuery(KNNEngine.LUCENE, ExpandNestedDocsQuery.class, VectorDataType.BYTE, true);
        testExpandNestedDocsQuery(KNNEngine.LUCENE, ExpandNestedDocsQuery.class, VectorDataType.FLOAT, true);
        testExpandNestedDocsQuery(KNNEngine.LUCENE, OSDiversifyingChildrenByteKnnVectorQuery.class, VectorDataType.BYTE, false);
        testExpandNestedDocsQuery(KNNEngine.LUCENE, OSDiversifyingChildrenFloatKnnVectorQuery.class, VectorDataType.FLOAT, false);
    }

    private void testExpandNestedDocsQuery(
        KNNEngine knnEngine,
        Class expectedQueryClass,
        VectorDataType vectorDataType,
        boolean expandNested
    ) {
        QueryShardContext queryShardContext = mock(QueryShardContext.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(queryShardContext.getParentFilter()).thenReturn(parentFilter);
        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .vector(testQueryVector)
            .byteVector(testByteQueryVector)
            .vectorDataType(vectorDataType)
            .k(testK)
            .expandNested(expandNested)
            .context(queryShardContext)
            .build();
        Query query = KNNQueryFactory.create(createQueryRequest);

        if (knnEngine == KNNEngine.LUCENE) {
            assertEquals(expectedQueryClass, ((LuceneEngineKnnVectorQuery) query).getLuceneQuery().getClass());
        } else {
            // Then
            assertEquals(expectedQueryClass, query.getClass());
        }
    }

    private int getLuceneK(final int k, final Map<String, ?> methodParameters) {
        return Math.max(k, IndexHyperParametersUtil.getHNSWEFSearchValue(methodParameters, testIndexName));
    }

    private Query buildLuceneFloatQuery(Map<String, ?> methodParams) {
        return KNNQueryFactory.create(
            BaseQueryFactory.CreateQueryRequest.builder()
                .knnEngine(KNNEngine.LUCENE)
                .vector(testQueryVector)
                .k(testK)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .methodParameters(methodParams)
                .vectorDataType(VectorDataType.FLOAT)
                .build()
        );
    }

    private Query buildLuceneByteQuery(Map<String, ?> methodParams) {
        return KNNQueryFactory.create(
            BaseQueryFactory.CreateQueryRequest.builder()
                .knnEngine(KNNEngine.LUCENE)
                .byteVector(testByteQueryVector)
                .k(testK)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .methodParameters(methodParams)
                .vectorDataType(VectorDataType.BYTE)
                .build()
        );
    }
}
