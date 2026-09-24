/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.collect.Multimap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Covers nested k-NN search with {@code expand_nested_docs} on a field that has rescoring enabled, which is the
 * combination reported in <a href="https://github.com/opensearch-project/k-NN/issues/3125">issue 3125</a>.
 *
 * A compression level of 4x resolves to the Lucene engine and turns rescoring on by default, so these tests
 * exercise the Lucene engine's two-phase search together with the nested expansion. The bug was that the final
 * reduction to k counted child documents rather than parent documents, so a single parent could consume the
 * whole budget and the remaining parents were dropped from the response.
 *
 * {@link ExpandNestedDocsIT} parameterizes over engines but only ever picks 1x or 32x compression, and 32x
 * resolves to Faiss, so this combination had no coverage there.
 */
public class ExpandNestedDocsWithRescoreIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "test-index-expand-nested-rescore";
    private static final String NESTED_FIELD = "nested_field";
    private static final String VECTOR_FIELD = "my_vector";
    private static final String NESTED_VECTOR_PATH = NESTED_FIELD + "." + VECTOR_FIELD;
    private static final String PARENT_FILTER_FIELD = "parking";
    private static final String CHILD_FILTER_FIELD = "storage";
    private static final String NESTED_FILTER_PATH = NESTED_FIELD + "." + CHILD_FILTER_FIELD;
    private static final int DIMENSION = 3;
    private static final int CHILDREN_PER_PARENT = 3;

    /**
     * The scenario straight out of the issue. Three parents of three children each, k=2, and the query vector is
     * an exact match for one of doc 1's children.
     *
     * Ground truth by l2 distance to [1,1,1]: doc 1 (best child [1,1,1]), then doc 3 (best child [9,9,9], 192),
     * then doc 2 (all children [10,10,10], 243). So the response must carry two parents, each with all three of
     * its children. The reported bug returned a single parent holding only two of its three children.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenRescoreEnabled_thenReturnKParentsWithAllChildren() {
        createNestedKnnIndex();
        indexDoc("1", new Object[][] { { 2, 2, 2 }, { 1, 1, 1 }, { 3, 3, 3 } });
        indexDoc("2", new Object[][] { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } });
        indexDoc("3", new Object[][] { { 9, 9, 9 }, { 200, 200, 200 }, { 300, 300, 300 } });
        refreshIndex(INDEX_NAME);
        // A single segment keeps per-leaf top-k effects out of the picture
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 2;
        String responseBody = search(k, new Object[] { 1, 1, 1 }, null);

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(responseBody, NESTED_FIELD);
        assertEquals(k, docIdToOffsets.keySet().size());
        for (String docId : docIdToOffsets.keySet()) {
            assertEquals("parent " + docId + " should carry all of its children", CHILDREN_PER_PARENT, docIdToOffsets.get(docId).size());
        }
        assertEquals(k, parseTotalSearchHits(responseBody));

        // Doc 3 beats doc 2 only on full precision distances (192 vs 243), so getting {1, 3} back in this order
        // is what proves the parent was selected on rescored vectors rather than on quantized ones.
        assertEquals(List.of("1", "3"), parseIds(responseBody));

        // The children have to be scored on full precision vectors too, otherwise a quantized child score leaks
        // into the parent through the nested score mode. These are the exact values a full precision index
        // returns: doc 1 averages 1.0, 0.25 and 0.0769, doc 3 averages 0.00518, 8.4e-6 and 3.7e-6.
        List<Double> scores = parseScores(responseBody);
        assertEquals(0.44230768d, scores.get(0), 1e-7);
        assertEquals(0.0017311643d, scores.get(1), 1e-9);
    }

    /**
     * Same shape, but with more parents than the rescore candidate pool can be satisfied by trivially, so the
     * reduction to k parents actually has to discard candidates. Every returned parent must still be complete.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenRescoreEnabledWithManyParents_thenReturnKParentsWithAllChildren() {
        createNestedKnnIndex();
        for (int i = 1; i <= 150; i++) {
            int base = i * 2;
            indexDoc(
                String.valueOf(i),
                new Object[][] { { base, base, base }, { base + 1, base + 1, base + 1 }, { base + 2, base + 2, base + 2 } }
            );
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 5;
        String responseBody = search(k, new Object[] { 0, 0, 0 }, null);

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(responseBody, NESTED_FIELD);
        assertEquals(k, docIdToOffsets.keySet().size());
        for (String docId : docIdToOffsets.keySet()) {
            assertEquals(CHILDREN_PER_PARENT, docIdToOffsets.get(docId).size());
        }
        assertEquals(k, parseTotalSearchHits(responseBody));
    }

    /**
     * Control: turning rescoring off from the query must not change the shape of the response. This isolates the
     * expansion from the rescore stage, so a failure here points at the plain expansion path rather than the fix.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenRescoreDisabledFromQuery_thenReturnKParentsWithAllChildren() {
        createNestedKnnIndex();
        indexDoc("1", new Object[][] { { 2, 2, 2 }, { 1, 1, 1 }, { 3, 3, 3 } });
        indexDoc("2", new Object[][] { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } });
        indexDoc("3", new Object[][] { { 9, 9, 9 }, { 200, 200, 200 }, { 300, 300, 300 } });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 2;
        String responseBody = search(k, new Object[] { 1, 1, 1 }, "false");

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(responseBody, NESTED_FIELD);
        assertEquals(k, docIdToOffsets.keySet().size());
        for (String docId : docIdToOffsets.keySet()) {
            assertEquals(CHILDREN_PER_PARENT, docIdToOffsets.get(docId).size());
        }
    }

    /**
     * An explicit oversample factor has to flow through to the candidate pool rather than being discarded
     * because the expansion is enabled.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenOversampleFactorProvided_thenReturnKParentsWithAllChildren() {
        createNestedKnnIndex();
        indexDoc("1", new Object[][] { { 2, 2, 2 }, { 1, 1, 1 }, { 3, 3, 3 } });
        indexDoc("2", new Object[][] { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } });
        indexDoc("3", new Object[][] { { 9, 9, 9 }, { 200, 200, 200 }, { 300, 300, 300 } });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 2;
        String responseBody = search(k, new Object[] { 1, 1, 1 }, "{ \"oversample_factor\": 3.0 }");

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(responseBody, NESTED_FIELD);
        assertEquals(k, docIdToOffsets.keySet().size());
        for (String docId : docIdToOffsets.keySet()) {
            assertEquals(CHILDREN_PER_PARENT, docIdToOffsets.get(docId).size());
        }
        assertEquals("1", parseIds(responseBody).get(0));
    }

    /**
     * A filter on the parent document has to survive the rescore stage: the filtered out parent must not come back
     * even though it is the closest one, and the parents that do come back must still carry all of their children.
     *
     * Doc 4 is an exact match for the query vector, so it would win outright without the filter. With the filter
     * applied the ground truth falls back to {1, 3} and the scores are the ones the unfiltered case produces,
     * which is what shows the filter narrowed the candidate pool without disturbing the full precision rescoring.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenRescoreEnabledWithParentFilter_thenReturnKMatchingParentsWithAllChildren() {
        createNestedKnnIndex();
        indexDocWithParentFilter("1", new Object[][] { { 2, 2, 2 }, { 1, 1, 1 }, { 3, 3, 3 } }, true);
        indexDocWithParentFilter("2", new Object[][] { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } }, true);
        indexDocWithParentFilter("3", new Object[][] { { 9, 9, 9 }, { 200, 200, 200 }, { 300, 300, 300 } }, true);
        indexDocWithParentFilter("4", new Object[][] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } }, false);
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 2;
        String filter = String.format(Locale.ROOT, "{\"term\":{\"%s\":true}}", PARENT_FILTER_FIELD);
        String responseBody = search(k, new Object[] { 1, 1, 1 }, null, false, filter);

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(responseBody, NESTED_FIELD);
        assertEquals(k, docIdToOffsets.keySet().size());
        assertFalse("the filtered out parent must not be returned", docIdToOffsets.containsKey("4"));
        for (String docId : docIdToOffsets.keySet()) {
            assertEquals("parent " + docId + " should carry all of its children", CHILDREN_PER_PARENT, docIdToOffsets.get(docId).size());
        }
        assertEquals(k, parseTotalSearchHits(responseBody));

        assertEquals(List.of("1", "3"), parseIds(responseBody));

        List<Double> scores = parseScores(responseBody);
        assertEquals(0.44230768d, scores.get(0), 1e-7);
        assertEquals(0.0017311643d, scores.get(1), 1e-9);
    }

    /**
     * A filter on the child documents has to narrow the expansion as well as the candidate selection. Only the
     * children matching the filter may be expanded into the inner hits, and the parents have to be ranked on their
     * matching children alone.
     *
     * Ground truth over the filtered children only: doc 1 keeps [1,1,1] and [3,3,3], doc 2 keeps all three
     * [10,10,10], doc 3 keeps only [200,200,200] and [300,300,300]. So doc 1 wins, doc 2 is second, and doc 3
     * drops out even though its unfiltered best child [9,9,9] would have beaten all of doc 2's.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenRescoreEnabledWithChildFilter_thenReturnOnlyMatchingChildren() {
        createNestedKnnIndex();
        indexDocWithChildFilter("1", new Object[][] { { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } }, new boolean[] { true, false, true });
        indexDocWithChildFilter("2", new Object[][] { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } }, new boolean[] { true, true, true });
        indexDocWithChildFilter(
            "3",
            new Object[][] { { 9, 9, 9 }, { 200, 200, 200 }, { 300, 300, 300 } },
            new boolean[] { false, true, true }
        );
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 2;
        String filter = String.format(
            Locale.ROOT,
            "{\"nested\":{\"path\":\"%s\",\"query\":{\"term\":{\"%s\":true}}}}",
            NESTED_FIELD,
            NESTED_FILTER_PATH
        );
        String responseBody = search(k, new Object[] { 1, 1, 1 }, null, false, filter);

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(responseBody, NESTED_FIELD);
        assertEquals(List.of("1", "2"), parseIds(responseBody));
        assertEquals(k, parseTotalSearchHits(responseBody));

        // Only the children that pass the filter get expanded, so doc 1 comes back without its middle child
        assertEquals(2, docIdToOffsets.get("1").size());
        assertTrue(docIdToOffsets.get("1").containsAll(List.of(0, 2)));
        assertEquals(CHILDREN_PER_PARENT, docIdToOffsets.get("2").size());

        // The excluded child must not contribute to its parent's score either. Averaged over the matching children
        // on full precision vectors, doc 1 is (1.0 + 1/13) / 2 and doc 2 is 1/244.
        List<Double> scores = parseScores(responseBody);
        assertEquals(0.53846154d, scores.get(0), 1e-7);
        assertEquals(0.0040983607d, scores.get(1), 1e-9);
    }

    /**
     * The two exact search stages must report their time under the Profile API. This is the only test that can
     * catch a missing registration in {@link org.opensearch.knn.plugin.KNNPlugin#getQueryProfileMetricsProvider}:
     * without it the breakdown has no {@code exact_search} metric at all, and asking for its timer throws.
     */
    @SneakyThrows
    public void testExpandNestedDocs_whenProfileEnabled_thenReportExactSearchTime() {
        createNestedKnnIndex();
        indexDoc("1", new Object[][] { { 2, 2, 2 }, { 1, 1, 1 }, { 3, 3, 3 } });
        indexDoc("2", new Object[][] { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } });
        indexDoc("3", new Object[][] { { 9, 9, 9 }, { 200, 200, 200 }, { 300, 300, 300 } });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 2;
        String responseBody = search(k, new Object[] { 1, 1, 1 }, null, true);

        // Profiling must not disturb the results
        assertEquals(List.of("1", "3"), parseIds(responseBody));

        // The query sits several levels down the profile tree, under the nested query's block join, so scan for
        // the breakdown at any depth rather than assuming a fixed nesting.
        List<Long> timings = collectBreakdownMetric(responseBody, "exact_search");
        assertFalse("expand_nested_docs with rescoring should report exact_search timings", timings.isEmpty());
        assertTrue("at least one exact_search timing should be non-zero, got " + timings, timings.stream().anyMatch(t -> t > 0L));

        // The collapsing rescore and the child expansion each run once per weight creation, and a single search
        // request creates the weight more than once, so assert that the two stages stay paired rather than
        // pinning an exact call count.
        List<Long> counts = collectBreakdownMetric(responseBody, "exact_search_count");
        long maxCount = counts.stream().mapToLong(Long::longValue).max().orElse(0L);
        assertTrue("expected at least one exact_search per stage, got " + counts, maxCount >= 2L);
        assertEquals("the collapsing rescore and the child expansion should be counted in pairs, got " + counts, 0L, maxCount % 2L);
    }

    /**
     * Collects one metric out of every {@code breakdown} object in the profile tree, at any depth. The k-NN query
     * is not at a fixed level of the tree, so walking is more robust than a hardcoded path.
     */
    @SuppressWarnings("unchecked")
    private List<Long> collectBreakdownMetric(final String responseBody, final String metric) throws Exception {
        Map<String, Object> parsed = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        List<Long> values = new ArrayList<>();
        Deque<Object> pending = new ArrayDeque<>();
        pending.push(parsed);
        while (pending.isEmpty() == false) {
            Object current = pending.pop();
            if (current instanceof Map<?, ?> map) {
                Object breakdown = map.get("breakdown");
                if (breakdown instanceof Map<?, ?> breakdownMap && breakdownMap.get(metric) instanceof Number value) {
                    values.add(value.longValue());
                }
                ((Map<String, Object>) map).values().forEach(pending::push);
            } else if (current instanceof List<?> list) {
                list.forEach(pending::push);
            }
        }
        return values;
    }

    /**
     * Creates the index from the issue: a nested knn_vector with 4x compression on disk, which resolves to the
     * Lucene engine and enables rescoring by default.
     */
    private void createNestedKnnIndex() throws Exception {
        String mapping = String.format(
            Locale.ROOT,
            "{\"properties\":{\"%s\":{\"type\":\"nested\",\"properties\":{\"%s\":"
                + "{\"type\":\"knn_vector\",\"dimension\":%d,\"space_type\":\"l2\","
                + "\"mode\":\"on_disk\",\"compression_level\":\"4x\"},"
                + "\"%s\":{\"type\":\"boolean\"}}},"
                + "\"%s\":{\"type\":\"boolean\"}}}",
            NESTED_FIELD,
            VECTOR_FIELD,
            DIMENSION,
            CHILD_FILTER_FIELD,
            PARENT_FILTER_FIELD
        );
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            // Force the graph to be built so the query goes through approximate search plus rescoring
            .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void indexDoc(final String docId, final Object[][] vectors) throws Exception {
        addKnnDoc(INDEX_NAME, docId, NestedKnnDocBuilder.create(NESTED_FIELD).addVectors(VECTOR_FIELD, vectors).build());
    }

    /**
     * Indexes a parent holding {@code vectors} plus a top level field the k-NN filter can select it by.
     */
    private void indexDocWithParentFilter(final String docId, final Object[][] vectors, final boolean filterValue) throws Exception {
        addKnnDoc(
            INDEX_NAME,
            docId,
            NestedKnnDocBuilder.create(NESTED_FIELD)
                .addVectors(VECTOR_FIELD, vectors)
                .addTopLevelField(PARENT_FILTER_FIELD, filterValue)
                .build()
        );
    }

    /**
     * Indexes a parent whose children each carry their own filter value, so a nested filter can select a subset of
     * the children of a single parent.
     */
    private void indexDocWithChildFilter(final String docId, final Object[][] vectors, final boolean[] filterValues) throws Exception {
        assert vectors.length == filterValues.length;
        NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(NESTED_FIELD);
        for (int i = 0; i < vectors.length; i++) {
            builder.addVectorWithMetadata(VECTOR_FIELD, vectors[i], CHILD_FILTER_FIELD, filterValues[i]);
        }
        addKnnDoc(INDEX_NAME, docId, builder.build());
    }

    /**
     * @param rescore raw json for the {@code rescore} clause, or null to leave it out and take the field default
     */
    private String search(final int k, final Object[] queryVector, final String rescore) throws Exception {
        return search(k, queryVector, rescore, false);
    }

    /**
     * @param rescore raw json for the {@code rescore} clause, or null to leave it out and take the field default
     * @param profile whether to ask for the Profile API breakdown
     */
    private String search(final int k, final Object[] queryVector, final String rescore, final boolean profile) throws Exception {
        return search(k, queryVector, rescore, profile, null);
    }

    /**
     * @param rescore raw json for the {@code rescore} clause, or null to leave it out and take the field default
     * @param profile whether to ask for the Profile API breakdown
     * @param filter  raw json for the k-NN query's {@code filter} clause, or null for an unfiltered search
     */
    private String search(final int k, final Object[] queryVector, final String rescore, final boolean profile, final String filter)
        throws Exception {
        String rescoreClause = rescore == null ? "" : String.format(Locale.ROOT, ",\"rescore\":%s", rescore);
        String filterClause = filter == null ? "" : String.format(Locale.ROOT, ",\"filter\":%s", filter);
        String body = String.format(
            Locale.ROOT,
            "{\"_source\":false,\"profile\":"
                + profile
                + ",\"size\":%d,\"query\":{\"nested\":{\"path\":\"%s\","
                + "\"query\":{\"knn\":{\"%s\":{\"vector\":%s,\"k\":%d,\"expand_nested_docs\":true%s%s}}},"
                + "\"inner_hits\":{\"size\":100,\"_source\":false}}}}",
            k,
            NESTED_FIELD,
            NESTED_VECTOR_PATH,
            java.util.Arrays.toString(queryVector),
            k,
            rescoreClause,
            filterClause
        );

        Request request = new Request("POST", String.format(Locale.ROOT, "/%s/_search", INDEX_NAME));
        request.setJsonEntity(body);
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return EntityUtils.toString(response.getEntity());
    }

    @Override
    public void tearDown() throws Exception {
        List<String> indices = List.of(INDEX_NAME);
        for (String index : indices) {
            try {
                deleteKNNIndex(index);
            } catch (Exception e) {
                // index may not exist for a test that failed during setup
            }
        }
        super.tearDown();
    }
}
