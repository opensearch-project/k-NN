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

import org.opensearch.action.ActionListener;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.indices.ModelMetadata.MODEL_METADATA_FIELD;

public class UpdateModelMetadataTransportActionTests extends KNNSingleNodeTestCase {
    //TODO: This maybe should operate as an integ test. Basically we need to validate all the functions in the
    // transport action. Really, I am only interested in the master operation. How do we:
    // 1. Verify that when the index doesn't exist, we fail gracefully
    // 2. Index exists, but custom metadata does not: make sure metadata is properly updated
    // 3. Removal request properly removes everything.
    // Basically, we want to check that this actually works.


    public void testPut() throws InterruptedException {
        // Setup the Model system index
        createIndex(MODEL_INDEX_NAME);

        // Setup the model
        String modelId = "test-model";
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 128);

        // Get update  transport action
        UpdateModelMetadataTransportAction updateModelMetadataTransportAction = node().injector()
                .getInstance(UpdateModelMetadataTransportAction.class);

        // Generate update request
        ClusterState clusterState = client().admin().cluster().prepareState().get().getState();
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest(modelId, false,
                modelMetadata);

        // Update the index metadata
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        updateModelMetadataTransportAction.masterOperation(
                updateModelMetadataRequest,
                clusterState,
                ActionListener.wrap(acknowledgedResponse -> {
                    assertTrue(acknowledgedResponse.isAcknowledged());

                    ClusterState updatedClusterState = client().admin().cluster().prepareState().get().getState();

                    IndexMetadata indexMetadata = updatedClusterState.metadata().index(MODEL_INDEX_NAME);
                    assertNotNull(indexMetadata);

                    Map<String, String> modelMetadataMap = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
                    assertNotNull(modelMetadataMap);

                    String modelAsString = modelMetadataMap.get(modelId);
                    assertNotNull(modelAsString);

                    ModelMetadata modelMetadataCopy = ModelMetadata.fromString(modelAsString);
                    assertEquals(modelMetadata, modelMetadataCopy);
                }, e -> fail("Update failed"))
        );

        assertTrue(inProgressLatch1.await(60, TimeUnit.SECONDS));

        // Add a new one
    }

    public void testRemove() {

    }
}
