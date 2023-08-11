/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.cluster.ClusterState;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.indices.ModelGraveyard;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

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
}
