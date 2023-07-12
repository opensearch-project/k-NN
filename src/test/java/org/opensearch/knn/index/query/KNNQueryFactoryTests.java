/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.mockito.Mockito;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;

public class KNNQueryFactoryTests extends KNNTestCase {
    private static final String FILTER_FILED_NAME = "foo";
    private static final String FILTER_FILED_VALUE = "fooval";
    private static final QueryBuilder FILTER_QUERY_BUILDER = new TermQueryBuilder(FILTER_FILED_NAME, FILTER_FILED_VALUE);
    private static final Query FILTER_QUERY = new TermQuery(new Term(FILTER_FILED_NAME, FILTER_FILED_VALUE));
    private final int testQueryDimension = 17;
    private final float[] testQueryVector = new float[testQueryDimension];
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int testK = 10;

    public void testCreateCustomKNNQuery() {
        for (KNNEngine knnEngine : KNNEngine.getEnginesThatCreateCustomSegmentFiles()) {
            Query query = KNNQueryFactory.create(
                knnEngine,
                testIndexName,
                testFieldName,
                testQueryVector,
                testK,
                DEFAULT_VECTOR_DATA_TYPE_FIELD
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
                knnEngine,
                testIndexName,
                testFieldName,
                testQueryVector,
                testK,
                DEFAULT_VECTOR_DATA_TYPE_FIELD
            );
            assertTrue(query.getClass().isAssignableFrom(KnnFloatVectorQuery.class));
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
            assertTrue(query.getClass().isAssignableFrom(KnnFloatVectorQuery.class));
        }
    }

    public void testCreateFaissQueryWithFilter_withValidValues_thenSuccess() {
        final KNNEngine knnEngine = KNNEngine.FAISS;
        final QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(testMapper.termQuery(Mockito.any(), Mockito.eq(mockQueryShardContext))).thenReturn(FILTER_QUERY);
        final KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .k(testK)
            .context(mockQueryShardContext)
            .filter(FILTER_QUERY_BUILDER)
            .build();
        final Query query = KNNQueryFactory.create(createQueryRequest);
        assertTrue(query instanceof KNNQuery);

        assertEquals(testIndexName, ((KNNQuery) query).getIndexName());
        assertEquals(testFieldName, ((KNNQuery) query).getField());
        assertEquals(testQueryVector, ((KNNQuery) query).getQueryVector());
        assertEquals(testK, ((KNNQuery) query).getK());
        assertEquals(FILTER_QUERY, ((KNNQuery) query).getFilterQuery());
    }
}
