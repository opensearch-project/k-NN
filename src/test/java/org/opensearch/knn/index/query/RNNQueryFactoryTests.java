/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.MAX_RESULTS_RADIAL_RESCORING;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

public class RNNQueryFactoryTests extends KNNTestCase {
    private static final String FILTER_FILED_NAME = "foo";
    private static final String FILTER_FILED_VALUE = "fooval";
    private static final QueryBuilder FILTER_QUERY_BUILDER = new TermQueryBuilder(FILTER_FILED_NAME, FILTER_FILED_VALUE);

    @Override
    public void setUp() throws Exception {
        super.setUp();
        RescoreRadialSearchQuery.initialize(new ExactSearcher(mock(ModelDao.OpenSearchKNNModelDao.class)));
    }

    private final int testQueryDimension = 17;
    private final float[] testQueryVector = new float[testQueryDimension];
    private final byte[] testByteQueryVector = new byte[testQueryDimension];
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final Float testRadius = 0.5f;
    private final int maxResultWindow = 20000;
    private final Map<String, ?> methodParameters = Map.of(METHOD_PARAMETER_EF_SEARCH, 100);

    public void testCreate_whenLucene_withRadiusQuery_withFloatVector() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            Query query = RNNQueryFactory.create(
                knnEngine,
                testIndexName,
                testFieldName,
                testQueryVector,
                testRadius,
                DEFAULT_VECTOR_DATA_TYPE_FIELD
            );
            assertEquals(RadialSearchQuery.class, query.getClass());
        }
    }

    public void testCreate_whenLucene_withRadiusQuery_withByteVector() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            MappedFieldType testMapper = mock(MappedFieldType.class);
            when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
            BitSetProducer parentFilter = mock(BitSetProducer.class);
            when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .byteVector(testByteQueryVector)
                .vectorDataType(VectorDataType.BYTE)
                .context(mockQueryShardContext)
                .filter(FILTER_QUERY_BUILDER)
                .build();
            Query query = RNNQueryFactory.create(createQueryRequest);
            assertEquals(RadialSearchQuery.class, query.getClass());
        }
    }

    public void testCreate_whenLucene_withFilter_thenSucceed() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            MappedFieldType testMapper = mock(MappedFieldType.class);
            when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .context(mockQueryShardContext)
                .filter(FILTER_QUERY_BUILDER)
                .radius(testRadius)
                .build();
            Query query = RNNQueryFactory.create(createQueryRequest);
            assertEquals(RadialSearchQuery.class, query.getClass());
        }
    }

    public void testCreate_whenFaiss_thenSucceed() {
        // Given
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(mockQueryShardContext.getIndexSettings().getMaxResultWindow()).thenReturn(maxResultWindow);

        final KNNQuery expectedQuery = KNNQuery.builder()
            .field(testFieldName)
            .queryVector(testQueryVector)
            .indexName(testIndexName)
            .radius(testRadius)
            .methodParameters(methodParameters)
            .context(new KNNQuery.Context(maxResultWindow))
            .build();

        // When
        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .methodParameters(methodParameters)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        // Then
        assertEquals(expectedQuery, query);
    }

    // Verify that Faiss radial search with 32x SQ wraps the inner KNNQuery in RescoreRadialSearchQuery.
    public void testCreate_whenFaissSQ32x_thenWrapsInRescoreRadialSearchQuery() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(indexSettings.getMaxResultWindow()).thenReturn(maxResultWindow);
        when(mockFieldType.isRescoringRequiredForRadial()).thenReturn(true);

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
        assertTrue(rescoreQuery.getInnerQuery() instanceof KNNQuery);
        assertEquals(testFieldName, rescoreQuery.getField());
        assertEquals(testRadius, rescoreQuery.getRadius(), 0.0f);
        // maxResultsSize should come from IndexSettings.getMaxResultWindow()
        assertEquals(maxResultWindow, rescoreQuery.getMaxResultsSize());
    }

    // Given: rescoring is required but no QueryShardContext is present
    // When: RNNQueryFactory creates the query
    // Then: maxResultsSize falls back to MAX_RESULTS_RADIAL_RESCORING
    public void testCreate_whenRescoringRequired_andNoContext_thenUsesDefaultMaxResultsSize() {
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockFieldType.isRescoringRequiredForRadial()).thenReturn(true);

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .vectorFieldType(mockFieldType)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
        assertEquals(MAX_RESULTS_RADIAL_RESCORING, rescoreQuery.getMaxResultsSize());
    }

    // Given: rescoring is required and IndexSettings has a custom maxResultWindow (e.g. 500)
    // When: RNNQueryFactory creates the query
    // Then: maxResultsSize is set to that custom value
    public void testCreate_whenRescoringRequired_andCustomMaxResultWindow_thenUsesCustomValue() {
        int customMaxResultWindow = 500;
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(indexSettings.getMaxResultWindow()).thenReturn(customMaxResultWindow);
        when(mockFieldType.isRescoringRequiredForRadial()).thenReturn(true);

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
        assertEquals(customMaxResultWindow, rescoreQuery.getMaxResultsSize());
    }

    // Verify that Lucene radial search with 32x SQ wraps the inner RadialSearchQuery
    // in RescoreRadialSearchQuery.
    public void testCreate_whenLuceneSQ32x_thenWrapsInRescoreRadialSearchQuery() {
        List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
            when(mockFieldType.isRescoringRequiredForRadial()).thenReturn(true);

            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .vectorFieldType(mockFieldType)
                .build();

            Query query = RNNQueryFactory.create(createQueryRequest);

            assertTrue(query instanceof RescoreRadialSearchQuery);
            RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
            assertTrue(rescoreQuery.getInnerQuery() instanceof RadialSearchQuery);
            assertEquals(testFieldName, rescoreQuery.getField());
            assertEquals(testRadius, rescoreQuery.getRadius(), 0.0f);
        }
    }

    // Verify that non-quantized Faiss radial search returns bare KNNQuery (no wrapper).
    public void testCreate_whenFaissNotQuantized_thenNoWrapper() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(indexSettings.getMaxResultWindow()).thenReturn(maxResultWindow);

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof KNNQuery);
        assertFalse(query instanceof RescoreRadialSearchQuery);
    }

    // Verify that non-quantized Lucene radial search returns bare RadialSearchQuery (no wrapper).
    public void testCreate_whenLuceneNotQuantized_thenNoWrapper() {
        List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .build();

            Query query = RNNQueryFactory.create(createQueryRequest);

            assertTrue(query instanceof RadialSearchQuery);
            assertFalse(query instanceof RescoreRadialSearchQuery);
        }
    }

    // Verify that createLuceneRadialQuery throws IllegalArgumentException for unsupported vector data types.
    // The default branch in the switch statement (line 146-152) should be hit when a BINARY vector type
    // is passed to the Lucene radial search path, since only FLOAT and BYTE are supported.
    public void testCreate_whenLuceneWithUnsupportedVectorDataType_thenThrows() {
        List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                // BINARY is not supported for Lucene radial search — should hit the default branch
                .vectorDataType(VectorDataType.BINARY)
                .build();

            expectThrows(IllegalArgumentException.class, () -> RNNQueryFactory.create(createQueryRequest));
        }
    }
}
