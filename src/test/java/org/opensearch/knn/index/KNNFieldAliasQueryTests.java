/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * Tests that a {@code knn} query issued against a field <em>alias</em> whose {@code path} points at a
 * {@code knn_vector} field behaves identically to the same query issued against the concrete field.
 *
 * <p>Regression coverage for the silent-zero-hits bug: a field alias exists only as mapping metadata and
 * never appears in Lucene's {@code FieldInfos}. {@code KNNQueryBuilder#doToQuery} resolved the mapping
 * through the alias correctly (so dimension / space type / engine were right) but then carried the
 * <em>alias</em> name into the {@code KNNQuery}. At the segment level {@code KNNWeight} looks the field name
 * up directly in {@code FieldInfos}, missed, and -- because a miss is not an error -- logged at DEBUG and
 * returned {@code EMPTY_TOPDOCS}. The user saw zero hits, no error, and nothing in the log.
 */
public class KNNFieldAliasQueryTests extends KNNSingleNodeTestCase {

    private static final String INDEX_NAME = "test-alias-index";
    private static final String CONCRETE_FIELD = "embedding_a";
    private static final String ALIAS_FIELD = "embedding";
    private static final int DIMENSION = 2;

    private static final Float[] VECTOR_1 = new Float[] { 1.0f, 1.0f };
    private static final Float[] VECTOR_2 = new Float[] { 2.0f, 2.0f };
    private static final Float[] VECTOR_3 = new Float[] { 3.0f, 3.0f };
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 1.0f };

    /**
     * Creates a mapping with a concrete knn_vector field plus an alias pointing at it.
     */
    private void createKnnMappingWithAlias(final String indexName, final KNNEngine engine) throws IOException {
        final PutMappingRequest request = new PutMappingRequest(indexName);
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(CONCRETE_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", engine.getName())
            .field("space_type", SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(ALIAS_FIELD)
            .field("type", "alias")
            .field("path", CONCRETE_FIELD)
            .endObject()
            .endObject()
            .endObject();
        request.source(builder);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    private void setUpIndex(final KNNEngine engine) throws IOException, InterruptedException, ExecutionException {
        createKNNIndex(INDEX_NAME);
        createKnnMappingWithAlias(INDEX_NAME, engine);
        addKnnDoc(INDEX_NAME, "1", CONCRETE_FIELD, VECTOR_1);
        addKnnDoc(INDEX_NAME, "2", CONCRETE_FIELD, VECTOR_2);
        addKnnDoc(INDEX_NAME, "3", CONCRETE_FIELD, VECTOR_3);
    }

    private long hits(final QueryBuilder query) {
        final SearchResponse response = client().prepareSearch(INDEX_NAME).setQuery(query).setSize(10).get();
        assertEquals(RestStatus.OK, response.status());
        return response.getHits().getTotalHits().value();
    }

    /**
     * Baseline: proves the alias itself is valid and resolves through the mapping. A non-vector query over
     * the alias must find all three docs. If this fails, the test setup is wrong rather than k-NN.
     */
    public void testExistsQueryOnAlias_thenResolves() throws IOException, InterruptedException, ExecutionException {
        setUpIndex(KNNEngine.FAISS);
        assertEquals(3L, hits(QueryBuilders.existsQuery(ALIAS_FIELD)));
    }

    /**
     * The core bug: an approximate k-NN query over the alias must return the same hits as over the concrete
     * field. Before the fix the alias variant silently returned zero hits.
     */
    public void testKnnQueryOnAlias_faiss_thenReturnsSameHitsAsConcreteField() throws IOException, InterruptedException,
        ExecutionException {
        setUpIndex(KNNEngine.FAISS);

        final long concreteHits = hits(new KNNQueryBuilder(CONCRETE_FIELD, QUERY_VECTOR, 3));
        assertEquals("sanity: concrete field must return all 3 docs", 3L, concreteHits);

        final long aliasHits = hits(new KNNQueryBuilder(ALIAS_FIELD, QUERY_VECTOR, 3));
        assertEquals("knn query over a field alias must return the same hits as the concrete field", concreteHits, aliasHits);
    }

    /**
     * Same as above for the Lucene engine, which takes a completely different code path
     * ({@code OSKnnFloatVectorQuery} rather than {@code KNNQuery}/{@code KNNWeight}).
     */
    public void testKnnQueryOnAlias_lucene_thenReturnsSameHitsAsConcreteField() throws IOException, InterruptedException,
        ExecutionException {
        setUpIndex(KNNEngine.LUCENE);

        final long concreteHits = hits(new KNNQueryBuilder(CONCRETE_FIELD, QUERY_VECTOR, 3));
        assertEquals("sanity: concrete field must return all 3 docs", 3L, concreteHits);

        final long aliasHits = hits(new KNNQueryBuilder(ALIAS_FIELD, QUERY_VECTOR, 3));
        assertEquals("lucene-engine knn query over a field alias must match the concrete field", concreteHits, aliasHits);
    }

    /**
     * Filtered k-NN over an alias. The filter is applied via a {@code FieldExistsQuery} on the query's field
     * name in {@code KNNQuery#getFilterWeight}, which is a second place an unresolved alias name would break.
     */
    public void testFilteredKnnQueryOnAlias_thenReturnsSameHitsAsConcreteField() throws IOException, InterruptedException,
        ExecutionException {
        setUpIndex(KNNEngine.FAISS);

        final long concreteHits = hits(new KNNQueryBuilder(CONCRETE_FIELD, QUERY_VECTOR, 3, QueryBuilders.existsQuery(CONCRETE_FIELD)));
        assertEquals("sanity: filtered concrete query must return all 3 docs", 3L, concreteHits);

        final KNNQueryBuilder alias = new KNNQueryBuilder(ALIAS_FIELD, QUERY_VECTOR, 3, QueryBuilders.existsQuery(CONCRETE_FIELD));
        assertEquals("filtered knn query over a field alias must match the concrete field", concreteHits, hits(alias));
    }

    /**
     * Radial search by {@code min_score} over an alias. This goes through {@code RNNQueryFactory} rather than
     * {@code KNNQueryFactory}, so it is a distinct field-name flow that needed fixing separately.
     */
    public void testRadialMinScoreQueryOnAlias_thenReturnsSameHitsAsConcreteField() throws IOException, InterruptedException,
        ExecutionException {
        setUpIndex(KNNEngine.FAISS);

        final KNNQueryBuilder concrete = KNNQueryBuilder.builder().fieldName(CONCRETE_FIELD).vector(QUERY_VECTOR).minScore(0.1f).build();
        final long concreteHits = hits(concrete);
        assertTrue("sanity: min_score radial query must match at least one doc", concreteHits > 0);

        final KNNQueryBuilder alias = KNNQueryBuilder.builder().fieldName(ALIAS_FIELD).vector(QUERY_VECTOR).minScore(0.1f).build();
        assertEquals("min_score radial query over a field alias must match the concrete field", concreteHits, hits(alias));
    }

    /**
     * Radial search by {@code max_distance} over an alias.
     */
    public void testRadialMaxDistanceQueryOnAlias_thenReturnsSameHitsAsConcreteField() throws IOException, InterruptedException,
        ExecutionException {
        setUpIndex(KNNEngine.FAISS);

        final KNNQueryBuilder concrete = KNNQueryBuilder.builder()
            .fieldName(CONCRETE_FIELD)
            .vector(QUERY_VECTOR)
            .maxDistance(100.0f)
            .build();
        final long concreteHits = hits(concrete);
        assertTrue("sanity: max_distance radial query must match at least one doc", concreteHits > 0);

        final KNNQueryBuilder alias = KNNQueryBuilder.builder().fieldName(ALIAS_FIELD).vector(QUERY_VECTOR).maxDistance(100.0f).build();
        assertEquals("max_distance radial query over a field alias must match the concrete field", concreteHits, hits(alias));
    }

    /**
     * {@code explain} over an alias must produce a populated explanation, not an empty/no-match one.
     */
    public void testExplainKnnQueryOnAlias_thenExplanationIsNotEmpty() throws IOException, InterruptedException, ExecutionException {
        setUpIndex(KNNEngine.FAISS);

        final SearchResponse response = client().prepareSearch(INDEX_NAME)
            .setQuery(new KNNQueryBuilder(ALIAS_FIELD, QUERY_VECTOR, 3))
            .setExplain(true)
            .setSize(10)
            .get();
        assertEquals(RestStatus.OK, response.status());
        assertTrue("explain over a field alias must return hits", response.getHits().getHits().length > 0);
        assertNotNull("explanation must be present for an alias query", response.getHits().getHits()[0].getExplanation());
    }
}
