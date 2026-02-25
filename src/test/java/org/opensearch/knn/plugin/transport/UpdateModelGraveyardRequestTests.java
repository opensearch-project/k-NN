/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import java.io.IOException;

public class UpdateModelGraveyardRequestTests extends KNNTestCase {

    public void testStreams() throws IOException {
        String modelId = "test-model-id";
        boolean isRemoveRequest = false;

        UpdateModelGraveyardRequest updateModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, isRemoveRequest);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        updateModelGraveyardRequest.writeTo(streamOutput);

        UpdateModelGraveyardRequest updateModelGraveyardRequest1 = new UpdateModelGraveyardRequest(streamOutput.bytes().streamInput());

        assertEquals(updateModelGraveyardRequest.getModelId(), updateModelGraveyardRequest1.getModelId());
        assertEquals(updateModelGraveyardRequest.isRemoveRequest(), updateModelGraveyardRequest1.isRemoveRequest());
    }

    public void testValidate() {
        String modelId = "test-model-id";
        UpdateModelGraveyardRequest updateModelGraveyardRequest1 = new UpdateModelGraveyardRequest(modelId, false);
        assertNull(updateModelGraveyardRequest1.validate());

        UpdateModelGraveyardRequest updateModelGraveyardRequest2 = new UpdateModelGraveyardRequest(modelId, true);
        assertNull(updateModelGraveyardRequest2.validate());

        UpdateModelGraveyardRequest updateModelGraveyardRequest3 = new UpdateModelGraveyardRequest("", false);
        assertNotNull(updateModelGraveyardRequest3.validate());

        UpdateModelGraveyardRequest updateModelGraveyardRequest4 = new UpdateModelGraveyardRequest("", true);
        assertNotNull(updateModelGraveyardRequest4.validate());
    }

    public void testGetModelId() {
        String modelId = "test-model-id";
        UpdateModelGraveyardRequest updateModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, false);

        assertEquals(modelId, updateModelGraveyardRequest.getModelId());
    }

    public void testIsRemoveRequest() {
        String modelId = "test-model-id";
        boolean isRemoveRequest = false;
        UpdateModelGraveyardRequest updateModelGraveyardRequest = new UpdateModelGraveyardRequest(modelId, isRemoveRequest);

        assertEquals(isRemoveRequest, updateModelGraveyardRequest.isRemoveRequest());
    }
}
