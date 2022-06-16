/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionListener;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.cluster.ClusterState;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.plugin.BlockedModelIds;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class UpdateBlockedModelTransportActionTests extends KNNSingleNodeTestCase {

    public void testExecutor() {
        UpdateBlockedModelTransportAction updateBlockedModelTransportAction = node().injector()
            .getInstance(UpdateBlockedModelTransportAction.class);
        assertEquals(ThreadPool.Names.SAME, updateBlockedModelTransportAction.executor());
    }

    public void testRead() throws IOException {
        UpdateBlockedModelTransportAction updateBlockedModelTransportAction = node().injector()
            .getInstance(UpdateBlockedModelTransportAction.class);
        AcknowledgedResponse acknowledgedResponse = new AcknowledgedResponse(true);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        acknowledgedResponse.writeTo(streamOutput);
        AcknowledgedResponse acknowledgedResponse1 = updateBlockedModelTransportAction.read(streamOutput.bytes().streamInput());

        assertEquals(acknowledgedResponse, acknowledgedResponse1);
    }

    public void testClusterManagerOperation() throws InterruptedException {

        String modelId = "test-model-id";

        // Get update transport action
        UpdateBlockedModelTransportAction updateBlockedModelTransportAction = node().injector()
            .getInstance(UpdateBlockedModelTransportAction.class);

        // Generate update request to add modelId to blocked list
        UpdateBlockedModelRequest addBlockedModelRequest = new UpdateBlockedModelRequest(modelId, false);

        // Get cluster state, update metadata, check cluster state - all asynchronously
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateBlockedModelTransportAction.masterOperation(
                addBlockedModelRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        BlockedModelIds blockedModelIds = updatedClusterState.metadata().custom(BlockedModelIds.TYPE);

                        assertNotNull(blockedModelIds);
                        assertEquals(1, blockedModelIds.size());
                        assertTrue(blockedModelIds.contains(modelId));

                        inProgressLatch1.countDown();

                    }, e -> fail("Update failed:" + e)));
                }, e -> fail("Update failed: " + e))
            );
        }, e -> fail("Update failed: " + e)));

        assertTrue(inProgressLatch1.await(60, TimeUnit.SECONDS));

        // Generate remove request to remove the modelId from blocked list
        UpdateBlockedModelRequest removeBlockedModelRequest = new UpdateBlockedModelRequest(modelId, true);

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateBlockedModelTransportAction.masterOperation(
                removeBlockedModelRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        BlockedModelIds blockedModelIds = updatedClusterState.metadata().custom(BlockedModelIds.TYPE);

                        assertNotNull(blockedModelIds);
                        assertEquals(0, blockedModelIds.size());
                        assertFalse(blockedModelIds.contains(modelId));

                        inProgressLatch2.countDown();
                    }, e -> fail("Update failed")));
                }, e -> fail("Update failed"))
            );
        }, e -> fail("Update failed")));

        assertTrue(inProgressLatch2.await(60, TimeUnit.SECONDS));
    }

    public void testCheckBlock() {
        UpdateBlockedModelTransportAction updateBlockedModelTransportAction = node().injector()
            .getInstance(UpdateBlockedModelTransportAction.class);
        assertNull(updateBlockedModelTransportAction.checkBlock(null, null));
    }
}
