/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.RequestOptions;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.KNNRestTestCase;

import static org.junit.Assume.assumeTrue;

/**
 * Exercises k-NN against Security's filtered reader and an unrestricted read role.
 */
public class FieldLevelSecurityIT extends KNNRestTestCase {
    private static final String RESTRICTED_ROLE = "knn_fls_it_restricted";
    private static final String UNRESTRICTED_ROLE = "knn_fls_it_unrestricted";
    private static final String PASSWORD = "KnnFlsIntegrationTest123!";
    private static final List<Integer> FIRST_VECTOR = List.of(1, 0, 0, 0, 0, 0, 0, 0);
    private static final List<Integer> SECOND_VECTOR = List.of(0, 1, 0, 0, 0, 0, 0, 0);

    @Before
    public void createSecurityRoles() throws Exception {
        assumeTrue("Security plugin is required for FLS tests", isHttps());
        put(
            "/_plugins/_security/api/roles/" + RESTRICTED_ROLE,
            Map.of(
                "cluster_permissions",
                List.of(),
                "index_permissions",
                List.of(
                    Map.of(
                        "index_patterns",
                        List.of("knn-fls-it-*"),
                        "allowed_actions",
                        List.of("read"),
                        "fls",
                        List.of(
                            "visible_faiss",
                            "visible_lucene",
                            "visible_disk",
                            "visible_float",
                            "visible_byte",
                            "visible_binary",
                            "visible_sq",
                            "visible_lucene_byte",
                            "items.visible_vec",
                            "tag"
                        )
                    )
                ),
                "tenant_permissions",
                List.of()
            )
        );
        put(
            "/_plugins/_security/api/roles/" + UNRESTRICTED_ROLE,
            Map.of(
                "cluster_permissions",
                List.of(),
                "index_permissions",
                List.of(Map.of("index_patterns", List.of("knn-fls-it-*"), "allowed_actions", List.of("read"))),
                "tenant_permissions",
                List.of()
            )
        );
        put(
            "/_plugins/_security/api/internalusers/" + RESTRICTED_ROLE,
            Map.of("password", PASSWORD, "opendistro_security_roles", List.of(RESTRICTED_ROLE))
        );
        put(
            "/_plugins/_security/api/internalusers/" + UNRESTRICTED_ROLE,
            Map.of("password", PASSWORD, "opendistro_security_roles", List.of(UNRESTRICTED_ROLE))
        );
    }

    @After
    public void removeSecurityRoles() throws IOException {
        if (isHttps()) {
            deleteIfPresent("/_plugins/_security/api/internalusers/" + RESTRICTED_ROLE);
            deleteIfPresent("/_plugins/_security/api/internalusers/" + UNRESTRICTED_ROLE);
            deleteIfPresent("/_plugins/_security/api/roles/" + RESTRICTED_ROLE);
            deleteIfPresent("/_plugins/_security/api/roles/" + UNRESTRICTED_ROLE);
        }
    }

    public void testFloatANNAndOuterBool() throws Exception {
        final String index = "knn-fls-it-float";
        createIndex(
            index,
            Map.of(
                "hidden_faiss",
                vectorField("faiss", "float", "l2"),
                "visible_faiss",
                vectorField("faiss", "float", "l2"),
                "hidden_lucene",
                vectorField("lucene", "float", "l2"),
                "visible_lucene",
                vectorField("lucene", "float", "l2"),
                "tag",
                Map.of("type", "keyword")
            ),
            Map.of()
        );
        putDoc(
            index,
            "1",
            Map.of(
                "hidden_faiss",
                FIRST_VECTOR,
                "visible_faiss",
                FIRST_VECTOR,
                "hidden_lucene",
                FIRST_VECTOR,
                "visible_lucene",
                FIRST_VECTOR,
                "tag",
                "match"
            )
        );
        putDoc(
            index,
            "2",
            Map.of(
                "hidden_faiss",
                SECOND_VECTOR,
                "visible_faiss",
                SECOND_VECTOR,
                "hidden_lucene",
                SECOND_VECTOR,
                "visible_lucene",
                SECOND_VECTOR,
                "tag",
                "match"
            )
        );

        assertHiddenPair(index, "hidden_faiss", "visible_faiss", SECOND_VECTOR, Map.of("k", 2));
        assertHiddenPair(index, "hidden_lucene", "visible_lucene", SECOND_VECTOR, Map.of("k", 2));

        final Map<String, Object> base = Map.of("must", Map.of("term", Map.of("tag", "match")));
        final Map<String, Object> withHiddenVector = Map.of(
            "must",
            base.get("must"),
            "should",
            knn("hidden_faiss", SECOND_VECTOR, Map.of("k", 2))
        );
        final String baseline = search(index, Map.of("size", 2, "query", Map.of("bool", base)), RESTRICTED_ROLE);
        final String hidden = search(index, Map.of("size", 2, "query", Map.of("bool", withHiddenVector)), RESTRICTED_ROLE);
        assertSameHitsAndScores(baseline, hidden);
        assertSameHitsAndScores(
            search(index, Map.of("size", 2, "query", Map.of("bool", withHiddenVector)), null),
            search(index, Map.of("size", 2, "query", Map.of("bool", withHiddenVector)), UNRESTRICTED_ROLE)
        );
    }

    public void testMaskedVectorsAndValueFetching() throws Exception {
        final String index = "knn-fls-it-masking";
        final List<String> fields = List.of("faiss_float", "lucene_float", "faiss_byte", "lucene_byte");
        createIndex(
            index,
            Map.of(
                "faiss_float",
                vectorField("faiss", "float", "l2"),
                "lucene_float",
                vectorField("lucene", "float", "l2"),
                "faiss_byte",
                vectorField("faiss", "byte", "l2"),
                "lucene_byte",
                vectorField("lucene", "byte", "l2"),
                "tag",
                Map.of("type", "keyword")
            ),
            Map.of()
        );
        putDoc(
            index,
            "1",
            Map.of(
                "faiss_float",
                FIRST_VECTOR,
                "lucene_float",
                FIRST_VECTOR,
                "faiss_byte",
                FIRST_VECTOR,
                "lucene_byte",
                FIRST_VECTOR,
                "tag",
                "match"
            )
        );
        for (List<String> maskedFields : List.of(fields, List.of("*"))) {
            final String role = "knn_fls_it_masked";
            try {
                createMaskedRole(role, maskedFields);
                for (String field : fields) {
                    for (Map<String, Object> options : List.of(
                        Map.<String, Object>of("k", 1),
                        Map.<String, Object>of("k", 1, "filter", Map.of("match_all", Map.of()))
                    )) {
                        final Map<String, Object> query = Map.of("size", 1, "query", knn(field, FIRST_VECTOR, options));
                        assertFalse(parseIds(search(index, query, UNRESTRICTED_ROLE)).isEmpty());
                        assertTrue(parseIds(search(index, query, role)).isEmpty());
                    }
                }
                assertVectorFetching(index, fields, role);
            } finally {
                deleteIfPresent("/_plugins/_security/api/internalusers/" + role);
                deleteIfPresent("/_plugins/_security/api/roles/" + role);
            }
        }
    }

    public void testMaskedBinaryBackedVectorFetching() throws Exception {
        final String index = "knn-fls-it-masking-flat";
        final List<String> fields = List.of("flat_float");
        put(
            "/" + index,
            Map.of(
                "settings",
                Map.of("index.knn", false, "number_of_shards", 1, "number_of_replicas", 0),
                "mappings",
                Map.of("properties", Map.of("flat_float", Map.of("type", "knn_vector", "dimension", 8)))
            )
        );
        putDoc(index, "1", Map.of("flat_float", FIRST_VECTOR));
        for (List<String> maskedFields : List.of(fields, List.of("*"))) {
            final String role = "knn_fls_it_masked_flat";
            try {
                createMaskedRole(role, maskedFields);
                assertVectorFetching(index, fields, role);
            } finally {
                deleteIfPresent("/_plugins/_security/api/internalusers/" + role);
                deleteIfPresent("/_plugins/_security/api/roles/" + role);
            }
        }
    }

    private void createMaskedRole(String role, List<String> fields) throws Exception {
        put(
            "/_plugins/_security/api/roles/" + role,
            Map.of(
                "cluster_permissions",
                List.of(),
                "index_permissions",
                List.of(Map.of("index_patterns", List.of("knn-fls-it-*"), "allowed_actions", List.of("read"), "masked_fields", fields)),
                "tenant_permissions",
                List.of()
            )
        );
        put("/_plugins/_security/api/internalusers/" + role, Map.of("password", PASSWORD, "opendistro_security_roles", List.of(role)));
    }

    private void assertVectorFetching(String index, List<String> fields, String maskedUser) throws Exception {
        for (String format : List.of("array", "binary")) {
            final List<Map<String, String>> requestedFields = fields.stream()
                .map(field -> Map.of("field", field, "format", format))
                .toList();
            final Map<String, Object> body = Map.of(
                "size",
                1,
                "_source",
                false,
                "query",
                Map.of("match_all", Map.of()),
                "docvalue_fields",
                requestedFields
            );
            for (String user : List.of(maskedUser, UNRESTRICTED_ROLE)) {
                final Map<String, Object> response = createParser(
                    MediaTypeRegistry.getDefaultMediaType().xContent(),
                    search(index, body, user)
                ).map();
                final List<?> hits = (List<?>) ((Map<?, ?>) response.get("hits")).get("hits");
                assertEquals(1, hits.size());
                final Map<?, ?> values = (Map<?, ?>) ((Map<?, ?>) hits.get(0)).get("fields");
                if (user.equals(maskedUser)) {
                    if (values != null) {
                        for (Object value : values.values()) {
                            assertTrue(((List<?>) value).isEmpty());
                        }
                    }
                } else {
                    assertNotNull(values);
                    assertEquals(fields.size(), values.size());
                    for (String field : fields) {
                        assertFalse(((List<?>) values.get(field)).isEmpty());
                    }
                }
            }
        }
    }

    public void testFilteredAndRadialANN() throws Exception {
        final String index = "knn-fls-it-filter";
        createIndex(
            index,
            Map.of(
                "hidden_faiss",
                vectorField("faiss", "float", "l2"),
                "visible_faiss",
                vectorField("faiss", "float", "l2"),
                "tag",
                Map.of("type", "keyword")
            ),
            Map.of()
        );
        putDoc(index, "1", Map.of("hidden_faiss", FIRST_VECTOR, "visible_faiss", FIRST_VECTOR, "tag", "match"));
        putDoc(index, "2", Map.of("hidden_faiss", SECOND_VECTOR, "visible_faiss", SECOND_VECTOR, "tag", "match"));

        assertHiddenPair(
            index,
            "hidden_faiss",
            "visible_faiss",
            FIRST_VECTOR,
            Map.of("k", 2, "filter", Map.of("term", Map.of("tag", "match")))
        );
        assertHiddenPair(index, "hidden_faiss", "visible_faiss", FIRST_VECTOR, Map.of("max_distance", 10));
    }

    public void testNestedAndOnDiskRescore() throws Exception {
        final String nestedIndex = "knn-fls-it-nested";
        createIndex(
            nestedIndex,
            Map.of(
                "items",
                Map.of(
                    "type",
                    "nested",
                    "properties",
                    Map.of("hidden_vec", vectorField("faiss", "float", "l2"), "visible_vec", vectorField("faiss", "float", "l2"))
                )
            ),
            Map.of()
        );
        putDoc(nestedIndex, "1", Map.of("items", List.of(Map.of("hidden_vec", FIRST_VECTOR, "visible_vec", FIRST_VECTOR))));
        putDoc(nestedIndex, "2", Map.of("items", List.of(Map.of("hidden_vec", SECOND_VECTOR, "visible_vec", SECOND_VECTOR))));
        assertHiddenPair(nestedIndex, "items.hidden_vec", "items.visible_vec", FIRST_VECTOR, Map.of("k", 2), true);

        final String diskIndex = "knn-fls-it-disk";
        final Map<String, Object> diskField = Map.of("type", "knn_vector", "dimension", 8, "space_type", "l2", "mode", "on_disk");
        createIndex(diskIndex, Map.of("hidden_disk", diskField, "visible_disk", diskField), Map.of());
        putDoc(diskIndex, "1", Map.of("hidden_disk", FIRST_VECTOR, "visible_disk", FIRST_VECTOR));
        putDoc(diskIndex, "2", Map.of("hidden_disk", SECOND_VECTOR, "visible_disk", SECOND_VECTOR));
        assertHiddenPair(diskIndex, "hidden_disk", "visible_disk", FIRST_VECTOR, Map.of("k", 2, "rescore", true));
        final String profile = search(
            diskIndex,
            Map.of("size", 2, "profile", true, "query", knn("visible_disk", FIRST_VECTOR, Map.of("k", 2, "rescore", true))),
            RESTRICTED_ROLE
        );
        final Map<String, Object> profileResponse = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), profile).map();
        assertTrue(sumMetric(profileResponse.get("profile"), "exact_search_count") > 0);
        assertTrue(sumMetric(profileResponse.get("profile"), "exact_search") > 0);
    }

    public void testByteBinaryQuantizedAndMemoryOptimizedVectors() throws Exception {
        for (boolean memoryOptimized : List.of(false, true)) {
            final String index = "knn-fls-it-variants-" + memoryOptimized;
            final Map<String, Object> fields = new HashMap<>();
            final Map<String, Object> floatField = vectorField("faiss", "float", "l2");
            final Map<String, Object> byteField = vectorField("faiss", "byte", "l2");
            final Map<String, Object> binaryField = vectorField("faiss", "binary", "hamming");
            final Map<String, Object> sqField = Map.of(
                "type",
                "knn_vector",
                "dimension",
                8,
                "data_type",
                "float",
                "method",
                Map.of(
                    "name",
                    "hnsw",
                    "engine",
                    "faiss",
                    "space_type",
                    "l2",
                    "parameters",
                    Map.of("ef_construction", 16, "m", 2, "encoder", Map.of("name", "sq", "parameters", Map.of("bits", 1)))
                )
            );
            final Map<String, Object> luceneByteField = vectorField("lucene", "byte", "l2");
            for (String scope : List.of("hidden", "visible")) {
                fields.put(scope + "_float", floatField);
                fields.put(scope + "_byte", byteField);
                fields.put(scope + "_binary", binaryField);
                fields.put(scope + "_sq", sqField);
                fields.put(scope + "_lucene_byte", luceneByteField);
            }
            createIndex(index, fields, Map.of("index.knn.memory_optimized_search", memoryOptimized, "index.use_compound_file", false));
            for (int docId = 0; docId < 8; docId++) {
                final List<Integer> vector = new ArrayList<>();
                for (int dimension = 0; dimension < 8; dimension++) {
                    vector.add(dimension == docId ? 1 : 0);
                }
                final int binary = docId == 7 ? -128 : 1 << docId;
                final Map<String, Object> document = new HashMap<>();
                for (String scope : List.of("hidden", "visible")) {
                    document.put(scope + "_float", vector);
                    document.put(scope + "_byte", vector);
                    document.put(scope + "_binary", List.of(binary));
                    document.put(scope + "_sq", vector);
                    document.put(scope + "_lucene_byte", vector);
                }
                putDoc(index, Integer.toString(docId), document);
            }
            for (String type : List.of("float", "byte", "sq", "lucene_byte")) {
                assertHiddenPair(index, "hidden_" + type, "visible_" + type, FIRST_VECTOR, Map.of("k", 2));
            }
            assertHiddenPair(index, "hidden_binary", "visible_binary", List.of(1), Map.of("k", 2));
            if (memoryOptimized) {
                final String explanation = search(
                    index,
                    Map.of("size", 2, "explain", true, "query", knn("visible_float", FIRST_VECTOR, Map.of("k", 2))),
                    RESTRICTED_ROLE
                );
                assertTrue(explanation.contains("memory optimized searching logic"));
            }
        }
    }

    private static long sumMetric(Object value, String key) {
        if (value instanceof Map<?, ?> map) {
            long sum = map.get(key) instanceof Number number ? number.longValue() : 0;
            for (Object child : map.values()) {
                sum += sumMetric(child, key);
            }
            return sum;
        }
        if (value instanceof List<?> list) {
            long sum = 0;
            for (Object child : list) {
                sum += sumMetric(child, key);
            }
            return sum;
        }
        return 0;
    }

    private void assertHiddenPair(String index, String hidden, String visible, List<Integer> vector, Map<String, Object> options)
        throws Exception {
        assertHiddenPair(index, hidden, visible, vector, options, false);
    }

    private void assertHiddenPair(
        String index,
        String hidden,
        String visible,
        List<Integer> vector,
        Map<String, Object> options,
        boolean nested
    ) throws Exception {
        final Map<String, Object> hiddenQuery = wrapQuery(knn(hidden, vector, options), nested);
        final Map<String, Object> visibleQuery = wrapQuery(knn(visible, vector, options), nested);
        final String admin = search(index, Map.of("size", 2, "query", hiddenQuery), null);
        final String unrestricted = search(index, Map.of("size", 2, "query", hiddenQuery), UNRESTRICTED_ROLE);
        final String restricted = search(index, Map.of("size", 2, "query", hiddenQuery), RESTRICTED_ROLE);
        final String visibleResult = search(index, Map.of("size", 2, "query", visibleQuery), RESTRICTED_ROLE);

        assertFalse(parseIds(admin).isEmpty());
        assertSameHitsAndScores(admin, unrestricted);
        assertTrue(parseIds(restricted).isEmpty());
        assertFalse(parseIds(visibleResult).isEmpty());
    }

    private static Map<String, Object> wrapQuery(Map<String, Object> query, boolean nested) {
        return nested ? Map.of("nested", Map.of("path", "items", "query", query)) : query;
    }

    private static Map<String, Object> knn(String field, List<Integer> vector, Map<String, Object> options) {
        final Map<String, Object> parameters = new HashMap<>(options);
        parameters.put("vector", vector);
        return Map.of("knn", Map.of(field, parameters));
    }

    private static Map<String, Object> vectorField(String engine, String dataType, String space) {
        return Map.of(
            "type",
            "knn_vector",
            "dimension",
            8,
            "data_type",
            dataType,
            "method",
            Map.of("name", "hnsw", "engine", engine, "space_type", space, "parameters", Map.of("ef_construction", 16, "m", 2))
        );
    }

    private void createIndex(String index, Map<String, Object> fields, Map<String, Object> extraSettings) throws Exception {
        final Map<String, Object> settings = new HashMap<>(extraSettings);
        settings.put("index.knn", true);
        settings.put("number_of_shards", 1);
        settings.put("number_of_replicas", 0);
        put("/" + index, Map.of("settings", settings, "mappings", Map.of("properties", fields)));
    }

    private void putDoc(String index, String id, Map<String, Object> document) throws Exception {
        put("/" + index + "/_doc/" + id + "?refresh=true", document);
    }

    private void put(String path, Map<String, Object> body) throws Exception {
        final Request request = new Request("PUT", path);
        request.setJsonEntity(XContentFactory.jsonBuilder().map(body).toString());
        adminClient().performRequest(request);
    }

    private String search(String index, Map<String, Object> body, String user) throws Exception {
        final Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(XContentFactory.jsonBuilder().map(body).toString());
        if (user != null) {
            final String credentials = Base64.getEncoder().encodeToString((user + ":" + PASSWORD).getBytes(StandardCharsets.UTF_8));
            request.setOptions(RequestOptions.DEFAULT.toBuilder().addHeader("Authorization", "Basic " + credentials).build());
        }
        final Response response = adminClient().performRequest(request);
        final String result = EntityUtils.toString(response.getEntity());
        @SuppressWarnings("unchecked")
        final Map<String, Object> shards = (Map<String, Object>) createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), result)
            .map()
            .get("_shards");
        assertEquals(0, ((Number) shards.get("failed")).intValue());
        return result;
    }

    private void assertSameHitsAndScores(String expected, String actual) throws IOException {
        assertEquals(parseIds(expected), parseIds(actual));
        assertEquals(parseScores(expected), parseScores(actual));
    }

    private void deleteIfPresent(String path) throws IOException {
        try {
            adminClient().performRequest(new Request("DELETE", path));
        } catch (ResponseException error) {
            if (error.getResponse().getStatusLine().getStatusCode() != 404) {
                throw error;
            }
        }
    }
}
