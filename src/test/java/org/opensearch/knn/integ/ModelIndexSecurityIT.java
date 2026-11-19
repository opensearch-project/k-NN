/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
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
import org.opensearch.knn.indices.ModelState;

import static org.junit.Assume.assumeTrue;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;

/**
 * Checks, with the Security plugin installed, that users who hold only the k-NN roles can train, read, search and
 * delete models. The model index and the internal model actions are reached by the plugin, never by the user.
 */
public class ModelIndexSecurityIT extends KNNRestTestCase {
    private static final String PASSWORD = "KnnModelIndexIntegTest123!";
    private static final String TRAINING_INDEX = "knn-model-it-training";
    private static final String TARGET_INDEX = "knn-model-it-target";
    private static final String FIELD = "vector";
    private static final int DIMENSION = 4;

    /** knn_full_access, plus read and scroll on the training index and create and write on the target index. */
    private static final String FULL = "knn_model_it_full";
    /** knn_read_access only. */
    private static final String READER = "knn_model_it_reader";
    /** Read and scroll on the training index and no k-NN actions. */
    private static final String NO_KNN = "knn_model_it_none";
    private static final String DATA_ROLE = "knn_model_it_data";

    @Before
    public void createUsers() throws Exception {
        assumeTrue("Security plugin is required for model index security tests", isHttps());
        put(
            "/_plugins/_security/api/roles/" + DATA_ROLE,
            Map.of(
                // Training reads the training index as the caller. The security plugin checks scroll, clear scroll
                // and the bulk behind a document write as cluster permissions.
                "cluster_permissions",
                List.of("indices:data/read/scroll*", "indices:data/write/bulk"),
                "index_permissions",
                List.of(
                    Map.of("index_patterns", List.of(TRAINING_INDEX), "allowed_actions", List.of("read")),
                    Map.of("index_patterns", List.of(TARGET_INDEX), "allowed_actions", List.of("crud", "create_index"))
                )
            )
        );
        createUser(FULL, List.of("knn_full_access", DATA_ROLE));
        createUser(READER, List.of("knn_read_access"));
        createUser(NO_KNN, List.of(DATA_ROLE));
        for (String user : List.of(FULL, READER, NO_KNN)) {
            // Each request below is sent with this user's credentials, never with the admin's.
            assertEquals(user, entityAsMap(as(user, "GET", "/_plugins/_security/authinfo", null)).get("user_name"));
        }
        createBasicKnnIndex(TRAINING_INDEX, FIELD, DIMENSION);
        bulkIngestRandomVectors(TRAINING_INDEX, FIELD, 40, DIMENSION);
    }

    @After
    public void deleteUsers() throws IOException {
        if (isHttps()) {
            for (String user : List.of(FULL, READER, NO_KNN)) {
                deleteIfPresent("/_plugins/_security/api/internalusers/" + user);
            }
            deleteIfPresent("/_plugins/_security/api/roles/" + DATA_ROLE);
        }
    }

    public void testModelLifecycleWithOnlyTheKnnRoles() throws Exception {
        String modelId = "knn-model-it-model";
        assertStatus(200, as(FULL, "POST", "/_plugins/_knn/models/" + modelId + "/_train", trainBody()));
        awaitModelCreated(modelId);

        // The model index is not readable for the user; only the k-NN APIs reach it.
        assertStatus(403, as(FULL, "GET", "/" + MODEL_INDEX_NAME + "/_doc/" + modelId, null));

        assertStatus(200, as(READER, "GET", "/_plugins/_knn/models/" + modelId, null));
        Response search = as(READER, "POST", "/_plugins/_knn/models/_search", "{\"query\":{\"match_all\":{}}}");
        assertStatus(200, search);
        assertTrue(body(search), body(search).contains("\"_id\":\"" + modelId + "\""));

        // Creating an index from the model reads the model as the plugin.
        String mapping = "{\"settings\":{\"index.knn\":true},\"mappings\":{\"properties\":{\""
            + FIELD
            + "\":{\"type\":\"knn_vector\",\"model_id\":\""
            + modelId
            + "\"}}}}";
        assertStatus(200, as(FULL, "PUT", "/" + TARGET_INDEX, mapping));
        assertStatus(201, as(FULL, "PUT", "/" + TARGET_INDEX + "/_doc/1?refresh=true", "{\"" + FIELD + "\":[1.0,2.0,3.0,4.0]}"));
        Response knn = as(
            FULL,
            "POST",
            "/" + TARGET_INDEX + "/_search",
            "{\"query\":{\"knn\":{\"" + FIELD + "\":{\"vector\":[1.0,2.0,3.0,4.0],\"k\":1}}}}"
        );
        assertStatus(200, knn);
        assertTrue(body(knn), body(knn).contains("\"_id\":\"1\""));
        adminClient().performRequest(new Request("DELETE", "/" + TARGET_INDEX));

        assertDenied(as(READER, "DELETE", "/_plugins/_knn/models/" + modelId, null), "cluster:admin/knn_delete_model_action");
        assertStatus(200, as(FULL, "DELETE", "/_plugins/_knn/models/" + modelId, null));
        assertStatus(404, as(FULL, "GET", "/_plugins/_knn/models/" + modelId, null));
    }

    public void testModelApisNeedAKnnRole() throws Exception {
        // The train API routes the job first, so the router action is the one denied.
        assertDenied(
            as(NO_KNN, "POST", "/_plugins/_knn/models/denied/_train", trainBody()),
            "cluster:admin/knn_training_job_router_action"
        );
        assertDenied(
            as(READER, "POST", "/_plugins/_knn/models/denied/_train", trainBody()),
            "cluster:admin/knn_training_job_router_action"
        );
        assertDenied(as(NO_KNN, "GET", "/_plugins/_knn/models/denied", null), "cluster:admin/knn_get_model_action");
        assertDenied(
            as(NO_KNN, "POST", "/_plugins/_knn/models/_search", "{\"query\":{\"match_all\":{}}}"),
            "cluster:admin/knn_search_model_action"
        );
        assertDenied(as(NO_KNN, "DELETE", "/_plugins/_knn/models/denied", null), "cluster:admin/knn_delete_model_action");
    }

    private String trainBody() throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .field("training_index", TRAINING_INDEX)
            .field("training_field", FIELD)
            .field("dimension", DIMENSION)
            .field("description", "model index security test")
            .field("method", xContentBuilderToMap(getModelMethodBuilder()))
            .endObject()
            .toString();
    }

    private void awaitModelCreated(String modelId) throws Exception {
        for (int attempt = 0; attempt < 60; attempt++) {
            Response response = as(FULL, "GET", "/_plugins/_knn/models/" + modelId, null);
            assertStatus(200, response);
            Map<String, Object> model = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), body(response)).map();
            ModelState state = ModelState.getModelState((String) model.get(MODEL_STATE));
            assertNotEquals("training failed: " + model, ModelState.FAILED, state);
            if (state == ModelState.CREATED) {
                return;
            }
            Thread.sleep(1000);
        }
        fail("model " + modelId + " was not created in time");
    }

    private void createUser(String name, List<String> roles) throws Exception {
        put("/_plugins/_security/api/internalusers/" + name, Map.of("password", PASSWORD, "opendistro_security_roles", roles));
    }

    private void put(String path, Map<String, Object> body) throws Exception {
        Request request = new Request("PUT", path);
        request.setJsonEntity(XContentFactory.jsonBuilder().map(body).toString());
        adminClient().performRequest(request);
    }

    private Response as(String user, String method, String path, String body) throws IOException {
        Request request = new Request(method, path);
        if (body != null) {
            request.setJsonEntity(body);
        }
        String credentials = Base64.getEncoder().encodeToString((user + ":" + PASSWORD).getBytes(StandardCharsets.UTF_8));
        request.setOptions(RequestOptions.DEFAULT.toBuilder().addHeader("Authorization", "Basic " + credentials).build());
        try {
            return adminClient().performRequest(request);
        } catch (ResponseException e) {
            return e.getResponse();
        }
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

    private static void assertDenied(Response response, String action) throws IOException {
        assertStatus(403, response);
        assertTrue(body(response), body(response).contains(action));
    }

    private static void assertStatus(int expected, Response response) throws IOException {
        assertEquals(body(response), expected, response.getStatusLine().getStatusCode());
    }

    private static String body(Response response) throws IOException {
        try {
            return response.getEntity() == null ? "" : EntityUtils.toString(response.getEntity());
        } catch (org.apache.hc.core5.http.ParseException e) {
            throw new IOException(e);
        }
    }
}
