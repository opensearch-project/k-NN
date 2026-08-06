/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.search.internal.SearchContext;

public class SizeBoundedRadialSearchQueryTests extends KNNTestCase {
    private static final String INDEX_NAME = "test-index";
    private static final String FIELD_NAME = "test-field";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f };

    @Override
    public void setUp() throws Exception {
        super.setUp();
        RescoreRadialSearchQuery.initialize(new ExactSearcher(mock(ModelDao.OpenSearchKNNModelDao.class)));
    }

    public void testCanApproximate_usesOversampledCandidateLimitAndRequestSizeFinalLimit() {
        SizeBoundedRadialSearchQuery query = new SizeBoundedRadialSearchQuery(createRequest(2.0f));
        SearchContext context = mock(SearchContext.class);
        when(context.size()).thenReturn(3);

        assertTrue(query.canApproximate(context));

        RescoreRadialSearchQuery resolvedQuery = query.getResolvedQuery();
        assertNotNull(resolvedQuery);
        assertEquals(6, resolvedQuery.getFirstPassK());
        assertEquals(3, resolvedQuery.getMaxResultsSize());
        assertFalse(resolvedQuery.getInnerQuery() instanceof RescoreKNNVectorQuery);
    }

    public void testCanApproximate_whenRequestSizeIsNotPositive_thenDeclinesApproximation() {
        SizeBoundedRadialSearchQuery query = new SizeBoundedRadialSearchQuery(createRequest(2.0f));
        SearchContext context = mock(SearchContext.class);
        when(context.size()).thenReturn(0);

        assertFalse(query.canApproximate(context));
        assertNull(query.getResolvedQuery());
    }

    public void testCanApproximate_capsOversampledCandidateLimit() {
        SizeBoundedRadialSearchQuery query = new SizeBoundedRadialSearchQuery(createRequest(2.0f));
        SearchContext context = mock(SearchContext.class);
        when(context.size()).thenReturn(RescoreContext.MAX_FIRST_PASS_RESULTS);

        assertTrue(query.canApproximate(context));

        RescoreRadialSearchQuery resolvedQuery = query.getResolvedQuery();
        assertEquals(RescoreContext.MAX_FIRST_PASS_RESULTS, resolvedQuery.getFirstPassK());
        assertEquals(RescoreContext.MAX_FIRST_PASS_RESULTS, resolvedQuery.getMaxResultsSize());
    }

    private static BaseQueryFactory.CreateQueryRequest createRequest(final float oversampleFactor) {
        return BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .vectorDataType(VectorDataType.FLOAT)
            .radius(0.5f)
            .rescoreContext(RescoreContext.builder().oversampleFactor(oversampleFactor).build())
            .build();
    }
}
