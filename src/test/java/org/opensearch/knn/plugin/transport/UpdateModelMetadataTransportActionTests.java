/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_METADATA_FIELD;

public class UpdateModelMetadataTransportActionTests extends KNNSingleNodeTestCase {

    public void testExecutor() {
        UpdateModelMetadataTransportAction updateModelMetadataTransportAction = node().injector()
            .getInstance(UpdateModelMetadataTransportAction.class);
        assertEquals(ThreadPool.Names.SAME, updateModelMetadataTransportAction.executor());
    }

    public void testRead() throws IOException {
        UpdateModelMetadataTransportAction updateModelMetadataTransportAction = node().injector()
            .getInstance(UpdateModelMetadataTransportAction.class);
        AcknowledgedResponse acknowledgedResponse = new AcknowledgedResponse(true);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        acknowledgedResponse.writeTo(streamOutput);
        AcknowledgedResponse acknowledgedResponse1 = updateModelMetadataTransportAction.read(streamOutput.bytes().streamInput());

        assertEquals(acknowledgedResponse, acknowledgedResponse1);
    }

    public void testClusterManagerOperation() throws InterruptedException {
        // Setup the Model system index
        createIndex(MODEL_INDEX_NAME);

        // Setup the model
        String modelId = "test-model";
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT
        );

        // Get update transport action
        UpdateModelMetadataTransportAction updateModelMetadataTransportAction = node().injector()
            .getInstance(UpdateModelMetadataTransportAction.class);

        // Generate update request
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest(modelId, false, modelMetadata);

        // Get cluster state, update metadata, check cluster state - all asynchronously
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelMetadataTransportAction.clusterManagerOperation(
                updateModelMetadataRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        IndexMetadata indexMetadata = updatedClusterState.metadata().index(MODEL_INDEX_NAME);
                        assertNotNull(indexMetadata);

                        Map<String, String> modelMetadataMap = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
                        assertNotNull(modelMetadataMap);

                        String modelAsString = modelMetadataMap.get(modelId);
                        assertNotNull(modelAsString);

                        ModelMetadata modelMetadataCopy = ModelMetadata.fromString(modelAsString);
                        assertEquals(modelMetadata, modelMetadataCopy);

                        inProgressLatch1.countDown();

                    }, e -> fail("Update failed:" + e)));
                }, e -> fail("Update failed: " + e))
            );
        }, e -> fail("Update failed: " + e)));

        assertTrue(inProgressLatch1.await(60, TimeUnit.SECONDS));

        // Generate remove request
        UpdateModelMetadataRequest removeModelMetadataRequest = new UpdateModelMetadataRequest(modelId, true, modelMetadata);

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse1 -> {
            ClusterState clusterState1 = stateResponse1.getState();
            updateModelMetadataTransportAction.clusterManagerOperation(
                removeModelMetadataRequest,
                clusterState1,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    client().admin().cluster().prepareState().execute(ActionListener.wrap(stateResponse2 -> {
                        ClusterState updatedClusterState = stateResponse2.getState();
                        IndexMetadata indexMetadata = updatedClusterState.metadata().index(MODEL_INDEX_NAME);
                        assertNotNull(indexMetadata);

                        Map<String, String> modelMetadataMap = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
                        assertNotNull(modelMetadataMap);

                        String modelAsString = modelMetadataMap.get(modelId);
                        assertNull(modelAsString);

                        inProgressLatch2.countDown();
                    }, e -> fail("Update failed")));
                }, e -> fail("Update failed"))
            );
        }, e -> fail("Update failed")));

        assertTrue(inProgressLatch2.await(60, TimeUnit.SECONDS));
    }

    public void testCheckBlock() {
        UpdateModelMetadataTransportAction updateModelMetadataTransportAction = node().injector()
            .getInstance(UpdateModelMetadataTransportAction.class);
        assertNull(updateModelMetadataTransportAction.checkBlock(null, null));
    }
}
