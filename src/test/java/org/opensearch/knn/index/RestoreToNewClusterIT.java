/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.hc.core5.http.HttpHost;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.RestClient;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;

/**
 * Restores a snapshot of k-NN's model index ({@code .opensearch-knn-models}) and a data index built from a trained
 * model onto a second, empty cluster, the way a customer recovers onto a new cluster. Checks that k-NN works there
 * with no further steps: the restored model and k-NN results match the source cluster, and the restored indices take
 * new documents and new models.
 *
 * <p>The empty cluster's address comes from {@code tests.second_cluster}. The
 * {@code integTestMultiCluster} task starts both clusters and sets it. The base class helpers all send to
 * {@code client()}, the source cluster, so steps on the empty cluster build their own requests.
 */
public class RestoreToNewClusterIT extends KNNRestTestCase {

    private static final String SECOND_CLUSTER = "tests.second_cluster";
    private static final String REPOSITORY = "restore-to-new-cluster";
    private static final String SNAPSHOT = "model-and-data";
    private static final String TRAINING_INDEX = "training-index";
    private static final String DATA_INDEX = "data-index";
    private static final String FIELD = "vector";
    private static final String MODEL_ID = "model-from-source";
    private static final String NEW_MODEL_ID = "model-trained-after-restore";
    private static final String NEW_DOC_ID = "written-after-restore";
    private static final String MODELS_URI = "/_plugins/_knn/models/";
    private static final int DIMENSIONS = 8;
    private static final int TRAINING_DOCS = 200;
    private static final int DATA_DOCS = 100;
    private static final int K = 5;
    private static final int TRAINING_ATTEMPTS = 30;
    private static final int TRAINING_DELAY_MILLIS = 1000;
    private static final Map<String, Object> MODEL_METHOD = Map.of(
        "name",
        "ivf",
        "engine",
        "faiss",
        "space_type",
        "l2",
        "parameters",
        Map.of("nlist", 4, "nprobes", 2)
    );

    private RestClient targetClient;

    @Before
    public void setUpTargetClient() throws IOException {
        String hosts = System.getProperty(SECOND_CLUSTER);
        assumeTrue(SECOND_CLUSTER + " is not set; run the integTestMultiCluster task", hosts != null);
        List<HttpHost> targetHosts = new ArrayList<>();
        for (String host : hosts.split(",")) {
            int portSeparator = host.lastIndexOf(':');
            targetHosts.add(buildHttpHost(host.substring(0, portSeparator), Integer.parseInt(host.substring(portSeparator + 1))));
        }
        targetClient = buildClient(restClientSettings(), targetHosts.toArray(new HttpHost[0]));
    }

    @After
    public void cleanUp() throws IOException {
        // The models are in use by the data index, so the data index goes first on both clusters.
        if (targetClient != null) {
            try {
                deleteIgnoringMissing(targetClient, "/" + DATA_INDEX);
                deleteIgnoringMissing(targetClient, MODELS_URI + MODEL_ID);
                deleteIgnoringMissing(targetClient, MODELS_URI + NEW_MODEL_ID);
                deleteIgnoringMissing(targetClient, "/" + TRAINING_INDEX);
                deleteIgnoringMissing(targetClient, "/_snapshot/" + REPOSITORY);
            } finally {
                targetClient.close();
            }
        }
        // The base classes then delete the source cluster's models, indices, snapshots and repositories.
        deleteIgnoringMissing(client(), "/" + DATA_INDEX);
    }

    public void testRestoreOntoNewCluster_whenModelIndexRestoredWithDataIndex_thenKnnWorks() throws Exception {
        // Source cluster: train a model, build a data index from it, and record what the source answers.
        createTrainingIndex(TRAINING_INDEX, createKnnIndexMapping(FIELD, DIMENSIONS));
        bulkAddKnnDocs(TRAINING_INDEX, FIELD, randomVectors(TRAINING_DOCS), TRAINING_DOCS);
        trainModel(MODEL_ID, TRAINING_INDEX, FIELD, DIMENSIONS, MODEL_METHOD, "trained on the source cluster");
        assertTrainingSucceeds(MODEL_ID, TRAINING_ATTEMPTS, TRAINING_DELAY_MILLIS);
        createKnnIndex(DATA_INDEX, createKnnIndexMapping(FIELD, MODEL_ID));
        bulkAddKnnDocs(DATA_INDEX, FIELD, randomVectors(DATA_DOCS), DATA_DOCS);
        float[][] queries = randomVectors(3);
        Map<String, Object> modelOnSource = model(client(), MODEL_ID);
        Map<String, Object> resultsOnSource = knnResults(client(), queries);
        resultsOnSource.forEach(
            (query, neighbors) -> assertEquals("neighbors of " + query + " on the source", K, ((List<?>) neighbors).size())
        );

        // Snapshot only the indices under test. createSnapshot would snapshot every index on the cluster.
        registerRepository(REPOSITORY, "fs", true, Settings.builder().put("location", repositoryLocation()).build());
        Request snapshot = new Request("PUT", "/_snapshot/" + REPOSITORY + "/" + SNAPSHOT);
        snapshot.addParameter("wait_for_completion", "true");
        snapshot.setJsonEntity(snapshotIndices());
        client().performRequest(snapshot);

        // The same write the target makes after its restore, so the target has an answer to match.
        float[] newVector = randomVectors(1)[0];
        addDocument(client(), newVector);
        float[][] queriesWithNewVector = Arrays.copyOf(queries, queries.length + 1);
        queriesWithNewVector[queries.length] = newVector;
        Map<String, Object> resultsOnSourceAfterWrite = knnResults(client(), queriesWithNewVector);

        // Target cluster: new, so it has never seen the model.
        Request getModel = new Request("GET", MODELS_URI + MODEL_ID);
        getModel.addParameter("ignore", "404");
        assertEquals("model on the new cluster", 404, targetClient.performRequest(getModel).getStatusLine().getStatusCode());

        registerReadOnlyRepository(targetClient);
        Request restore = new Request("POST", "/_snapshot/" + REPOSITORY + "/" + SNAPSHOT + "/_restore");
        restore.addParameter("wait_for_completion", "true");
        restore.setJsonEntity(snapshotIndices());
        Map<String, Object> restoreResponse = entityAsMap(targetClient.performRequest(restore));
        assertEquals(
            "restore had failed shards: " + restoreResponse,
            0,
            XContentMapValues.extractValue("snapshot.shards.failed", restoreResponse)
        );

        // The restored model and the data index built from it answer exactly as on the source.
        assertEquals("restored model", modelOnSource, model(targetClient, MODEL_ID));
        assertEquals("k-NN results after restore", resultsOnSource, knnResults(targetClient, queries));

        // The restored data index takes a new document, which needs the restored model.
        addDocument(targetClient, newVector);
        Map<String, Object> resultsOnTargetAfterWrite = knnResults(targetClient, queriesWithNewVector);
        assertEquals("k-NN results after a new document", resultsOnSourceAfterWrite, resultsOnTargetAfterWrite);
        List<?> neighborsOfNewVector = (List<?>) resultsOnTargetAfterWrite.get(Arrays.toString(newVector));
        assertEquals("nearest neighbor of the new vector", List.of(NEW_DOC_ID, 1.0), neighborsOfNewVector.get(0));

        // The restored model index takes a new model, trained from the restored training index.
        trainModel(targetClient, NEW_MODEL_ID);
        assertTrainingSucceeds(targetClient, NEW_MODEL_ID);
        assertEquals("models on the target", List.of(MODEL_ID, NEW_MODEL_ID), modelIds(targetClient));
    }

    private static String snapshotIndices() throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .field("indices", String.join(",", MODEL_INDEX_NAME, DATA_INDEX, TRAINING_INDEX))
            .field("include_global_state", false)
            .endObject()
            .toString();
    }

    private static String repositoryLocation() {
        return System.getProperty("tests.path.repo") + "/" + REPOSITORY;
    }

    private static void registerReadOnlyRepository(RestClient cluster) throws IOException {
        XContentBuilder body = XContentFactory.jsonBuilder()
            .startObject()
            .field("type", "fs")
            .startObject("settings")
            .field("location", repositoryLocation())
            .field("readonly", true)
            .endObject()
            .endObject();
        Request request = new Request("PUT", "/_snapshot/" + REPOSITORY);
        request.setJsonEntity(body.toString());
        cluster.performRequest(request);
    }

    private static void addDocument(RestClient cluster, float[] vector) throws IOException {
        Request request = new Request("PUT", "/" + DATA_INDEX + "/_doc/" + NEW_DOC_ID);
        request.addParameter("refresh", "true");
        request.setJsonEntity(XContentFactory.jsonBuilder().startObject().field(FIELD, vector).endObject().toString());
        cluster.performRequest(request);
    }

    private static void trainModel(RestClient cluster, String modelId) throws IOException {
        XContentBuilder body = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX)
            .field(TRAIN_FIELD_PARAMETER, FIELD)
            .field(DIMENSION, DIMENSIONS)
            .field(KNN_METHOD, MODEL_METHOD)
            .endObject();
        Request request = new Request("POST", MODELS_URI + modelId + "/_train");
        request.setJsonEntity(body.toString());
        cluster.performRequest(request);
    }

    /** Same as {@link #assertTrainingSucceeds(String, int, int)}, on the given cluster. */
    private static void assertTrainingSucceeds(RestClient cluster, String modelId) throws Exception {
        for (int attempt = 0; attempt < TRAINING_ATTEMPTS; attempt++) {
            Thread.sleep(TRAINING_DELAY_MILLIS);
            ModelState state = ModelState.getModelState((String) model(cluster, modelId).get(MODEL_STATE));
            if (state == ModelState.CREATED) {
                return;
            }
            assertNotEquals("state of " + modelId, ModelState.FAILED, state);
        }
        fail("Training of " + modelId + " did not succeed after " + TRAINING_ATTEMPTS + " attempts");
    }

    private static Map<String, Object> model(RestClient cluster, String modelId) throws IOException {
        return entityAsMap(cluster.performRequest(new Request("GET", MODELS_URI + modelId)));
    }

    private static List<String> modelIds(RestClient cluster) throws IOException {
        XContentBuilder body = XContentFactory.jsonBuilder()
            .startObject()
            .field("_source", false)
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject();
        Request request = new Request("POST", MODELS_URI + "_search");
        request.setJsonEntity(body.toString());
        List<?> hits = (List<?>) XContentMapValues.extractValue("hits.hits", entityAsMap(cluster.performRequest(request)));
        return hits.stream().map(hit -> (String) ((Map<?, ?>) hit).get("_id")).sorted().collect(Collectors.toList());
    }

    /** For each query vector: the ids and scores of its k nearest neighbors, in order. */
    private static Map<String, Object> knnResults(RestClient cluster, float[][] queries) throws IOException {
        Map<String, Object> results = new LinkedHashMap<>();
        for (float[] query : queries) {
            XContentBuilder body = XContentFactory.jsonBuilder().startObject().field("_source", false).startObject("query");
            KNNQueryBuilder.builder().fieldName(FIELD).vector(query).k(K).build().doXContent(body, ToXContent.EMPTY_PARAMS);
            body.endObject().endObject();
            Request request = new Request("POST", "/" + DATA_INDEX + "/_search");
            request.addParameter("size", Integer.toString(K));
            request.setJsonEntity(body.toString());
            List<?> hits = (List<?>) XContentMapValues.extractValue("hits.hits", entityAsMap(cluster.performRequest(request)));
            List<List<Object>> neighbors = hits.stream()
                .map(hit -> List.of(((Map<?, ?>) hit).get("_id"), ((Map<?, ?>) hit).get("_score")))
                .collect(Collectors.toList());
            results.put(Arrays.toString(query), neighbors);
        }
        return results;
    }

    private static float[][] randomVectors(int count) {
        float[][] vectors = new float[count][DIMENSIONS];
        for (float[] vector : vectors) {
            for (int d = 0; d < DIMENSIONS; d++) {
                vector[d] = randomFloat();
            }
        }
        return vectors;
    }

    private static void deleteIgnoringMissing(RestClient cluster, String endpoint) throws IOException {
        Request request = new Request("DELETE", endpoint);
        request.addParameter("ignore", "404");
        cluster.performRequest(request);
    }
}
