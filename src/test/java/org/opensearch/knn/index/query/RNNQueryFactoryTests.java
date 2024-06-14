/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

public class RNNQueryFactoryTests extends KNNTestCase {
    private static final String FILTER_FILED_NAME = "foo";
    private static final String FILTER_FILED_VALUE = "fooval";
    private static final QueryBuilder FILTER_QUERY_BUILDER = new TermQueryBuilder(FILTER_FILED_NAME, FILTER_FILED_VALUE);
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
            assertEquals(FloatVectorSimilarityQuery.class, query.getClass());
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
            assertEquals(ByteVectorSimilarityQuery.class, query.getClass());
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
            assertEquals(FloatVectorSimilarityQuery.class, query.getClass());
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
}
