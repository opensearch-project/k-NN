/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.admin.indices.create.CreateIndexRequestBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.cluster.ClusterState;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.exception.DeleteModelException;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelGraveyard;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.PROPERTIES;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

public class UpdateModelGraveyardTransportActionTests extends KNNSingleNodeTestCase {

    public void testExecutor() {
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction = node().injector()
            .getInstance(UpdateModelGraveyardTransportAction.class);
        assertEquals(ThreadPool.Names.SAME, updateModelGraveyardTransportAction.executor());
    }

    public void testRead() throws IOException {
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction = node().injector()
            .getInstance(UpdateModelGraveyardTransportAction.class);
        AcknowledgedResponse acknowledgedResponse = new AcknowledgedResponse(true);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        acknowledgedResponse.writeTo(streamOutput);
        AcknowledgedResponse acknowledgedResponse1 = updateModelGraveyardTransportAction.read(streamOutput.bytes().streamInput());

        assertEquals(acknowledgedResponse, acknowledgedResponse1);
    }

    public void testClusterManagerOperation() throws InterruptedException {

        String modelId = "test-model-id";

        // Get update transport action
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction = node().injector()
            .getInstance(UpdateModelGraveyardTransportAction.class);

        // Generate update request to add modelId to model graveyard
        UpdateModelGraveyardRequest addModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, false);

        // Get cluster state, update metadata, check cluster state - all asynchronously
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelGraveyardTransportAction.clusterManagerOperation(
                addModelGraveyardRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        ModelGraveyard modelGraveyard = updatedClusterState.metadata().custom(ModelGraveyard.TYPE);

                        assertNotNull(modelGraveyard);
                        assertEquals(1, modelGraveyard.size());
                        assertTrue(modelGraveyard.contains(modelId));

                        inProgressLatch1.countDown();

                    }, e -> fail("Update failed:" + e)));
                }, e -> fail("Update failed: " + e))
            );
        }, e -> fail("Update failed: " + e)));

        assertTrue(inProgressLatch1.await(60, TimeUnit.SECONDS));

        String modelId1 = "test-model-id-1";
        // Generate update request to add modelId1 to model graveyard
        UpdateModelGraveyardRequest addModelGraveyardRequest1 = new UpdateModelGraveyardRequest(modelId1, false);

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelGraveyardTransportAction.clusterManagerOperation(
                addModelGraveyardRequest1,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        ModelGraveyard modelGraveyard = updatedClusterState.metadata().custom(ModelGraveyard.TYPE);

                        assertNotNull(modelGraveyard);
                        assertEquals(2, modelGraveyard.size());
                        assertTrue(modelGraveyard.contains(modelId1));

                        ModelGraveyard modelGraveyardPrev = clusterState1.metadata().custom(ModelGraveyard.TYPE);
                        assertFalse(modelGraveyardPrev.contains(modelId1));

                        // Assertions to validate ModelGraveyard Diff
                        ModelGraveyard.ModelGraveyardDiff diff = new ModelGraveyard.ModelGraveyardDiff(modelGraveyardPrev, modelGraveyard);
                        assertEquals(0, diff.getRemoved().size());
                        assertEquals(1, diff.getAdded().size());
                        assertTrue(diff.getAdded().contains(modelId1));

                        ModelGraveyard updatedModelGraveyard = diff.apply(modelGraveyardPrev);
                        assertEquals(2, updatedModelGraveyard.size());
                        assertTrue(updatedModelGraveyard.contains(modelId));
                        assertTrue(updatedModelGraveyard.contains(modelId1));

                        inProgressLatch2.countDown();
                    }, e -> fail("Update failed")));
                }, e -> fail("Update failed"))
            );
        }, e -> fail("Update failed")));

        assertTrue(inProgressLatch2.await(60, TimeUnit.SECONDS));

        // Generate remove request to remove the modelId from model graveyard
        UpdateModelGraveyardRequest removeModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, true);

        final CountDownLatch inProgressLatch3 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelGraveyardTransportAction.clusterManagerOperation(
                removeModelGraveyardRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        ModelGraveyard modelGraveyard = updatedClusterState.metadata().custom(ModelGraveyard.TYPE);

                        assertNotNull(modelGraveyard);
                        assertEquals(1, modelGraveyard.size());
                        assertFalse(modelGraveyard.contains(modelId));

                        ModelGraveyard modelGraveyardPrev = clusterState1.metadata().custom(ModelGraveyard.TYPE);
                        assertTrue(modelGraveyardPrev.contains(modelId));

                        // Assertions to validate ModelGraveyard Diff
                        ModelGraveyard.ModelGraveyardDiff diff = new ModelGraveyard.ModelGraveyardDiff(modelGraveyardPrev, modelGraveyard);
                        assertEquals(1, diff.getRemoved().size());
                        assertEquals(0, diff.getAdded().size());
                        assertTrue(diff.getRemoved().contains(modelId));

                        ModelGraveyard updatedModelGraveyard = diff.apply(modelGraveyardPrev);
                        assertEquals(1, updatedModelGraveyard.size());
                        assertFalse(updatedModelGraveyard.contains(modelId));
                        assertTrue(updatedModelGraveyard.contains(modelId1));

                        inProgressLatch3.countDown();
                    }, e -> fail("Update failed")));
                }, e -> fail("Update failed"))
            );
        }, e -> fail("Update failed")));

        assertTrue(inProgressLatch3.await(60, TimeUnit.SECONDS));
    }

    public void testCheckBlock() {
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction = node().injector()
            .getInstance(UpdateModelGraveyardTransportAction.class);
        assertNull(updateModelGraveyardTransportAction.checkBlock(null, null));
    }

    public void testClusterManagerOperation_GetIndicesUsingModel() throws IOException, ExecutionException, InterruptedException {
        // Get update transport action
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction = node().injector()
            .getInstance(UpdateModelGraveyardTransportAction.class);

        String modelId = "test-model-id";
        byte[] modelBlob = "testModel".getBytes();
        int dimension = 2;

        createIndex(MODEL_INDEX_NAME);

        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                dimension,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            modelBlob,
            modelId
        );

        // created model and added it to index
        addModel(model);

        // Create basic index (not using k-NN)
        String testIndex1 = "test-index1";
        createIndex(testIndex1);

        // Attempt to add model id to graveyard with one non-knn index present, should succeed
        UpdateModelGraveyardRequest addModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, false);
        updateModelGraveyardAndAssertNoError(updateModelGraveyardTransportAction, addModelGraveyardRequest);

        // Remove model from graveyard to prepare for next check
        UpdateModelGraveyardRequest removeModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, true);
        updateModelGraveyardAndAssertNoError(updateModelGraveyardTransportAction, removeModelGraveyardRequest);

        // Create k-NN index not using the model
        String testIndex2 = "test-index2";
        createKNNIndex(testIndex2);

        // Attempt to add model id to graveyard with one non-knn index and one k-nn index not using model present, should succeed
        updateModelGraveyardAndAssertNoError(updateModelGraveyardTransportAction, addModelGraveyardRequest);

        // Remove model from graveyard to prepare for next check
        updateModelGraveyardAndAssertNoError(updateModelGraveyardTransportAction, removeModelGraveyardRequest);

        // Create k-NN index using model
        String testIndex3 = "test-index3";
        String testField3 = "test-field3";

        /*
            Constructs the following json:
            {
              "properties": {
                "test-field3": {
                  "type": "knn_vector",
                  "model_id": "test-model-id"
                }
              }
            }
         */
        XContentBuilder mappings3 = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(testField3)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(MODEL_ID, modelId)
            .endObject()
            .endObject()
            .endObject();

        XContentBuilder settings = XContentFactory.jsonBuilder().startObject().field(TestUtils.INDEX_KNN, "true").endObject();

        CreateIndexRequestBuilder createIndexRequestBuilder3 = client().admin()
            .indices()
            .prepareCreate(testIndex3)
            .setMapping(mappings3)
            .setSettings(settings);
        createIndex(testIndex3, createIndexRequestBuilder3);

        // Attempt to add model id to graveyard when one index is using model, should fail
        List<String> indicesUsingModel = new ArrayList<>();
        indicesUsingModel.add(testIndex3);
        updateModelGraveyardAndAssertDeleteModelException(
            updateModelGraveyardTransportAction,
            addModelGraveyardRequest,
            indicesUsingModel.toString()
        );

        // Create second k-NN index using model
        String testIndex4 = "test-index4";
        String testField4 = "test-field4";
        String standardField = "standard-field";

        /*
            Constructs the following json:
            {
              "properties": {
                "standard-field": {
                  "type": "knn_vector",
                  "dimension": "2"
                }
                "test-field4": {
                  "type": "knn_vector",
                  "model_id": "test-model-id"
                }
              }
            }
         */
        XContentBuilder mappings4 = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(standardField)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(testField4)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(MODEL_ID, modelId)
            .endObject()
            .endObject()
            .endObject();

        CreateIndexRequestBuilder createIndexRequestBuilder4 = client().admin()
            .indices()
            .prepareCreate(testIndex4)
            .setMapping(mappings4)
            .setSettings(settings);
        createIndex(testIndex4, createIndexRequestBuilder4);

        // Add index at beginning to match order of list returned by getIndicesUsingModel()
        indicesUsingModel.add(0, testIndex4);

        // Attempt to add model id to graveyard when one index is using model, should fail
        updateModelGraveyardAndAssertDeleteModelException(
            updateModelGraveyardTransportAction,
            addModelGraveyardRequest,
            indicesUsingModel.toString()
        );
    }

    private void updateModelGraveyardAndAssertNoError(
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction,
        UpdateModelGraveyardRequest updateModelGraveyardRequest
    ) throws InterruptedException {
        final CountDownLatch countDownLatch = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelGraveyardTransportAction.clusterManagerOperation(
                updateModelGraveyardRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());
                    countDownLatch.countDown();
                }, e -> { fail("Update failed: " + e); })
            );
        }, e -> fail("Update failed: " + e)));
        assertTrue(countDownLatch.await(60, TimeUnit.SECONDS));
    }

    private void updateModelGraveyardAndAssertDeleteModelException(
        UpdateModelGraveyardTransportAction updateModelGraveyardTransportAction,
        UpdateModelGraveyardRequest updateModelGraveyardRequest,
        String indicesPresentInException
    ) throws InterruptedException {
        final CountDownLatch countDownLatch = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelGraveyardTransportAction.clusterManagerOperation(
                updateModelGraveyardRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    fail();
                }, e -> {
                    assertTrue(e instanceof DeleteModelException);
                    assertEquals(
                        String.format(
                            "Cannot delete model [%s].  Model is in use by the following indices %s, which must be deleted first.",
                            updateModelGraveyardRequest.getModelId(),
                            indicesPresentInException
                        ),
                        e.getMessage()
                    );
                    countDownLatch.countDown();
                })
            );
        }, e -> fail("Update failed: " + e)));

        assertTrue(countDownLatch.await(60, TimeUnit.SECONDS));
    }
}
