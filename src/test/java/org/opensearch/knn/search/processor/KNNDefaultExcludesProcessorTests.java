/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.fetch.subphase.FetchSourceContext;

import java.util.LinkedHashSet;
import java.util.Set;

public class KNNDefaultExcludesProcessorTests extends KNNTestCase {

    private KNNDefaultExcludesProcessor processor;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        processor = new KNNDefaultExcludesProcessor("test-tag", "test-desc", false);
    }

    public void testType() {
        assertEquals("knn_default_excludes", processor.getType());
    }

    public void testProcessRequest_nullSource_returnsUnmodified() {
        SearchRequest request = new SearchRequest();
        SearchRequest result = processor.processRequest(request);
        assertNull(result.source());
    }

    public void testProcessRequest_nullQuery_returnsUnmodified() {
        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder());
        SearchRequest result = processor.processRequest(request);
        assertNull(result.source().fetchSource());
    }

    public void testProcessRequest_nonKnnQuery_noExcludes() {
        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder().query(new MatchAllQueryBuilder()));
        SearchRequest result = processor.processRequest(request);
        assertNull(result.source().fetchSource());
    }

    public void testProcessRequest_knnQuery_addsExclude() {
        KNNQueryBuilder knnQuery = KNNQueryBuilder.builder().fieldName("my_vector").vector(new float[] { 1.0f, 2.0f, 3.0f }).k(5).build();

        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder().query(knnQuery));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertTrue(ctx.fetchSource());
        assertEquals(0, ctx.includes().length);
        assertArrayEquals(new String[] { "my_vector" }, ctx.excludes());
    }

    public void testProcessRequest_preservesUserIncludes() {
        KNNQueryBuilder knnQuery = KNNQueryBuilder.builder().fieldName("my_vector").vector(new float[] { 1.0f, 2.0f }).k(3).build();

        SearchRequest request = new SearchRequest();
        request.source(
            new SearchSourceBuilder().query(knnQuery).fetchSource(new FetchSourceContext(true, new String[] { "title" }, new String[0]))
        );

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertArrayEquals(new String[] { "title" }, ctx.includes());
        assertArrayEquals(new String[] { "my_vector" }, ctx.excludes());
    }

    public void testProcessRequest_userIncludesVectorField_noExclude() {
        KNNQueryBuilder knnQuery = KNNQueryBuilder.builder().fieldName("my_vector").vector(new float[] { 1.0f, 2.0f }).k(3).build();

        SearchRequest request = new SearchRequest();
        request.source(
            new SearchSourceBuilder().query(knnQuery).fetchSource(new FetchSourceContext(true, new String[] { "my_vector" }, new String[0]))
        );

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertArrayEquals(new String[] { "my_vector" }, ctx.includes());
        assertEquals(0, ctx.excludes().length);
    }

    public void testProcessRequest_boolQueryWithKnn_addsExclude() {
        KNNQueryBuilder knnQuery = KNNQueryBuilder.builder().fieldName("embedding").vector(new float[] { 0.1f, 0.2f }).k(10).build();

        BoolQueryBuilder boolQuery = new BoolQueryBuilder().must(knnQuery).filter(new MatchAllQueryBuilder());

        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder().query(boolQuery));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertArrayEquals(new String[] { "embedding" }, ctx.excludes());
    }

    public void testProcessRequest_preservesExistingExcludes() {
        KNNQueryBuilder knnQuery = KNNQueryBuilder.builder().fieldName("vec").vector(new float[] { 1.0f }).k(1).build();

        SearchRequest request = new SearchRequest();
        request.source(
            new SearchSourceBuilder().query(knnQuery)
                .fetchSource(new FetchSourceContext(true, new String[0], new String[] { "internal_field" }))
        );

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        Set<String> excludes = Set.of(ctx.excludes());
        assertTrue(excludes.contains("internal_field"));
        assertTrue(excludes.contains("vec"));
    }

    public void testCollectKnnFieldNames_nestedBool() {
        KNNQueryBuilder knn1 = KNNQueryBuilder.builder().fieldName("f1").vector(new float[] { 1.0f }).k(1).build();
        KNNQueryBuilder knn2 = KNNQueryBuilder.builder().fieldName("f2").vector(new float[] { 2.0f }).k(1).build();

        BoolQueryBuilder inner = new BoolQueryBuilder().should(knn2);
        BoolQueryBuilder outer = new BoolQueryBuilder().must(knn1).should(inner);

        Set<String> fields = new LinkedHashSet<>();
        KNNDefaultExcludesProcessor.collectKnnFieldNames(outer, fields);

        assertTrue(fields.contains("f1"));
        assertTrue(fields.contains("f2"));
        assertEquals(2, fields.size());
    }
}
