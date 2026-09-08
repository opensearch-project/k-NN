/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Verifies that a {@code knn} query stored in a {@code percolator} field scores the percolated document.
 * <p>
 * Percolating a single document evaluates the stored queries against a Lucene {@code MemoryIndex}, which is
 * not backed by a segment, so the query falls back to exact search. Percolating several documents in one
 * request builds a real in-memory index instead, so both paths are covered here and asserted to agree.
 */
public class PercolatorIT extends KNNRestTestCase {

    private static final String INDEX = "knn-percolator";
    private static final String CONTROL_INDEX = "knn-percolator-control";
    private static final String QUERY_FIELD = "query";
    private static final String VECTOR_FIELD = "target_field";
    private static final String FILTER_FIELD = "category";
    private static final int DIMENSION = 3;
    private static final float[] SIMILAR_VECTOR = { 1.0f, 2.0f, 3.01f };
    private static final float[] DISSIMILAR_VECTOR = { -3.0f, 1.0f, -0.5f };

    @SneakyThrows
    public void testKnnInPercolator_thenScoresMatchNormalSearch() {
        createPercolatorIndex(SpaceType.COSINESIMIL, 1);
        storeQuery("sub", knnQuery(null, null));

        createKnnIndex(
            CONTROL_INDEX,
            getKNNDefaultIndexSettings(),
            createKnnIndexMapping(VECTOR_FIELD, DIMENSION, METHOD_HNSW, KNNEngine.FAISS.getName(), SpaceType.COSINESIMIL.getValue())
        );
        addKnnDoc(CONTROL_INDEX, "1", VECTOR_FIELD, SIMILAR_VECTOR);
        Response controlResponse = searchKNNIndex(CONTROL_INDEX, knnSearchBody(), 10);
        float expected = parseSearchResponseScore(EntityUtils.toString(controlResponse.getEntity()), VECTOR_FIELD).get(0);

        Map<String, Object> percolated = percolate(List.of(SIMILAR_VECTOR), null);
        assertEquals(0, shardFailures(percolated));
        assertEquals(1, totalHits(percolated));
        assertEquals(expected, firstScore(percolated), 0.0001d);
    }

    /**
     * Space type is resolved from the field mapping. A {@code MemoryIndex} leaf carries no field attributes,
     * so a field whose recorded Lucene similarity function does not match its space type, which is every
     * binary field, would otherwise be scored with L2.
     */
    @SneakyThrows
    public void testKnnInPercolator_whenBinaryField_thenScoresWithHamming() {
        Request create = new Request("PUT", "/" + INDEX);
        create.setJsonEntity(
            "{\"settings\":{\"index\":{\"knn\":true,\"number_of_shards\":1,\"number_of_replicas\":0}},"
                + "\"mappings\":{\"properties\":{\""
                + QUERY_FIELD
                + "\":{\"type\":\"percolator\"},"
                + "\""
                + VECTOR_FIELD
                + "\":{\"type\":\"knn_vector\",\"dimension\":8,\"data_type\":\"binary\","
                + "\"method\":{\"name\":\"hnsw\",\"space_type\":\"hamming\",\"engine\":\"faiss\"}}}}}"
        );
        client().performRequest(create);
        storeQuery("sub", "{\"knn\":{\"" + VECTOR_FIELD + "\":{\"vector\":[7],\"k\":5}}}");

        Request search = new Request("POST", "/" + INDEX + "/_search");
        search.setJsonEntity("{\"query\":{\"percolate\":{\"field\":\"" + QUERY_FIELD + "\",\"document\":{\"" + VECTOR_FIELD + "\":[3]}}}}");
        Map<String, Object> response = asMap(client().performRequest(search));

        assertEquals(0, shardFailures(response));
        assertEquals(1, totalHits(response));
        // Hamming distance between 7 and 3 is one bit, so 1 / (1 + 1). Scoring with L2 would give 1 / (1 + 16).
        assertEquals(0.5d, firstScore(response), 0.0001d);
    }

    @SneakyThrows
    public void testKnnInPercolator_whenMinScoreNotMet_thenNoMatch() {
        createPercolatorIndex(SpaceType.COSINESIMIL, 1);
        storeQuery("sub", knnQuery(0.9f, null));

        assertEquals(1, totalHits(percolate(List.of(SIMILAR_VECTOR), null)));

        Map<String, Object> excluded = percolate(List.of(DISSIMILAR_VECTOR), null);
        assertEquals(0, shardFailures(excluded));
        assertEquals(0, totalHits(excluded));
    }

    @SneakyThrows
    public void testKnnInPercolator_whenFilterExcludesDocument_thenNoMatch() {
        createPercolatorIndex(SpaceType.COSINESIMIL, 1);
        storeQuery("sub", knnQuery(null, "news"));

        assertEquals(1, totalHits(percolate(List.of(SIMILAR_VECTOR), "news")));

        Map<String, Object> excluded = percolate(List.of(SIMILAR_VECTOR), "sports");
        assertEquals(0, shardFailures(excluded));
        assertEquals(0, totalHits(excluded));
    }

    @SneakyThrows
    public void testKnnInPercolator_whenDocumentHasNoVector_thenNoMatchAndNoFailure() {
        createPercolatorIndex(SpaceType.COSINESIMIL, 1);
        storeQuery("sub", knnQuery(null, null));

        Request request = new Request("POST", "/" + INDEX + "/_search");
        request.setJsonEntity(
            "{\"query\":{\"percolate\":{\"field\":\"" + QUERY_FIELD + "\",\"document\":{\"" + FILTER_FIELD + "\":\"news\"}}}}"
        );
        Map<String, Object> response = asMap(client().performRequest(request));

        assertEquals(0, shardFailures(response));
        assertEquals(0, totalHits(response));
    }

    /**
     * Percolating several documents builds a real in-memory index instead of a {@code MemoryIndex}. That
     * index has segments but no native engine files, so it already reached exact search through
     * {@code isMissingNativeEngineFiles}. This pins the new single document path to the score that path
     * already returns.
     */
    @SneakyThrows
    public void testKnnInPercolator_whenBatched_thenAgreesWithSingleDocument() {
        createPercolatorIndex(SpaceType.COSINESIMIL, 1);
        storeQuery("sub", knnQuery(null, null));

        Map<String, Object> single = percolate(List.of(SIMILAR_VECTOR), null);
        Map<String, Object> batched = percolate(List.of(SIMILAR_VECTOR, SIMILAR_VECTOR), null);

        assertEquals(0, shardFailures(batched));
        assertEquals(1, totalHits(batched));
        assertEquals(firstScore(single), firstScore(batched), 0.0001d);
    }

    /**
     * Memory optimized search hands radial search a threshold on the {@code MAXIMUM_INNER_PRODUCT} scale for
     * {@code cosinesimil}, while exact search scores with the cosine function. Percolation only has the exact
     * path, so an unconverted threshold would reject every document.
     */
    @SneakyThrows
    public void testKnnInPercolator_whenMemoryOptimizedRadial_thenMatches() {
        Request create = new Request("PUT", "/" + INDEX);
        create.setJsonEntity(
            "{\"settings\":{\"index\":{\"knn\":true,\"number_of_shards\":1,\"number_of_replicas\":0}},"
                + "\"mappings\":{\"properties\":{\""
                + QUERY_FIELD
                + "\":{\"type\":\"percolator\"},"
                + "\""
                + VECTOR_FIELD
                + "\":{\"type\":\"knn_vector\",\"dimension\":"
                + DIMENSION
                + ",\"space_type\":\"cosinesimil\",\"mode\":\"on_disk\",\"compression_level\":\"1x\"}}}}"
        );
        client().performRequest(create);
        storeQuery("sub", knnQuery(0.9f, null));

        Map<String, Object> response = percolate(List.of(new float[] { 1.0f, 2.0f, 3.0f }), null);
        assertEquals(0, shardFailures(response));
        assertEquals(1, totalHits(response));
        // The stored query vector and the percolated document vector are identical, so cosine scores 1.0.
        assertEquals(1.0d, firstScore(response), 0.0001d);
    }

    /**
     * Percolation runs per shard. Before the fallback existed, a shard holding a {@code knn} query threw and
     * the failure was swallowed into partial results, leaving the caller with HTTP 200 and zero hits.
     */
    @SneakyThrows
    public void testKnnInPercolator_withMultipleShards_thenNoShardFailures() {
        createPercolatorIndex(SpaceType.COSINESIMIL, 5);
        for (int i = 0; i < 10; i++) {
            storeQuery("sub-" + i, knnQuery(null, null));
        }

        Map<String, Object> response = percolate(List.of(SIMILAR_VECTOR), null);
        assertEquals(0, shardFailures(response));
        assertEquals(10, totalHits(response));
    }

    private void createPercolatorIndex(SpaceType spaceType, int shards) throws Exception {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(QUERY_FIELD)
            .field("type", "percolator")
            .endObject()
            .startObject(FILTER_FIELD)
            .field("type", "keyword")
            .endObject()
            .startObject(VECTOR_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", spaceType.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Request request = new Request("PUT", "/" + INDEX);
        request.setJsonEntity(
            "{\"settings\":{\"index\":{\"knn\":true,\"number_of_shards\":"
                + shards
                + ",\"number_of_replicas\":0}},\"mappings\":"
                + mapping.toString()
                + "}"
        );
        client().performRequest(request);
    }

    private String knnQuery(Float minScore, String filterCategory) {
        StringBuilder query = new StringBuilder("{\"knn\":{\"" + VECTOR_FIELD + "\":{\"vector\":[1.0,2.0,3.0],");
        query.append(minScore == null ? "\"k\":5" : "\"min_score\":" + minScore);
        if (filterCategory != null) {
            query.append(",\"filter\":{\"term\":{\"").append(FILTER_FIELD).append("\":\"").append(filterCategory).append("\"}}");
        }
        return query.append("}}}").toString();
    }

    private String knnSearchBody() {
        return "{\"query\":{\"knn\":{\"" + VECTOR_FIELD + "\":{\"vector\":[1.0,2.0,3.0],\"k\":5}}}}";
    }

    private void storeQuery(String id, String query) throws Exception {
        Request request = new Request("PUT", "/" + INDEX + "/_doc/" + id + "?refresh=true");
        request.setJsonEntity("{\"" + QUERY_FIELD + "\":" + query + "}");
        client().performRequest(request);
    }

    private Map<String, Object> percolate(List<float[]> vectors, String category) throws Exception {
        List<String> documents = new ArrayList<>();
        for (float[] vector : vectors) {
            StringBuilder document = new StringBuilder("{\"" + VECTOR_FIELD + "\":[");
            for (int i = 0; i < vector.length; i++) {
                document.append(i == 0 ? "" : ",").append(vector[i]);
            }
            document.append("]");
            if (category != null) {
                document.append(",\"").append(FILTER_FIELD).append("\":\"").append(category).append("\"");
            }
            documents.add(document.append("}").toString());
        }

        // A single document exercises the MemoryIndex path, several exercise the real in-memory index.
        String target = documents.size() == 1 ? "\"document\":" + documents.get(0) : "\"documents\":[" + String.join(",", documents) + "]";
        Request request = new Request("POST", "/" + INDEX + "/_search");
        request.setJsonEntity("{\"query\":{\"percolate\":{\"field\":\"" + QUERY_FIELD + "\"," + target + "}}}");
        return asMap(client().performRequest(request));
    }

    private Map<String, Object> asMap(Response response) throws Exception {
        return createParser(JsonXContent.jsonXContent, EntityUtils.toString(response.getEntity())).map();
    }

    @SuppressWarnings("unchecked")
    private int totalHits(Map<String, Object> response) {
        Map<String, Object> hits = (Map<String, Object>) response.get("hits");
        return (int) ((Map<String, Object>) hits.get("total")).get("value");
    }

    @SuppressWarnings("unchecked")
    private int shardFailures(Map<String, Object> response) {
        return (int) ((Map<String, Object>) response.get("_shards")).get("failed");
    }

    @SuppressWarnings("unchecked")
    private double firstScore(Map<String, Object> response) {
        Map<String, Object> hits = (Map<String, Object>) response.get("hits");
        List<Map<String, Object>> hitList = (List<Map<String, Object>>) hits.get("hits");
        assertFalse("expected at least one hit", hitList.isEmpty());
        return ((Number) hitList.get(0).get("_score")).doubleValue();
    }
}
