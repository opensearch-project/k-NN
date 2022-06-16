/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import java.io.IOException;

public class UpdateBlockedModelRequestTests extends KNNTestCase {

    public void testStreams() throws IOException {
        String modelId = "test-model-id";
        boolean isRemoveRequest = false;

        UpdateBlockedModelRequest updateBlockedModelRequest = new UpdateBlockedModelRequest(modelId, isRemoveRequest);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        updateBlockedModelRequest.writeTo(streamOutput);

        UpdateBlockedModelRequest updateBlockedModelRequest1 = new UpdateBlockedModelRequest(streamOutput.bytes().streamInput());

        assertEquals(updateBlockedModelRequest.getModelId(), updateBlockedModelRequest1.getModelId());
        assertEquals(updateBlockedModelRequest.isRemoveRequest(), updateBlockedModelRequest1.isRemoveRequest());
    }

    public void testValidate() {
        String modelId = "test-model-id";
        UpdateBlockedModelRequest updateBlockedModelRequest1 = new UpdateBlockedModelRequest(modelId, false);
        assertNull(updateBlockedModelRequest1.validate());

        UpdateBlockedModelRequest updateBlockedModelRequest2 = new UpdateBlockedModelRequest(modelId, true);
        assertNull(updateBlockedModelRequest2.validate());

        UpdateBlockedModelRequest updateBlockedModelRequest3 = new UpdateBlockedModelRequest("", false);
        assertNotNull(updateBlockedModelRequest3.validate());

        UpdateBlockedModelRequest updateBlockedModelRequest4 = new UpdateBlockedModelRequest("", true);
        assertNotNull(updateBlockedModelRequest4.validate());
    }

    public void testGetModelId() {
        String modelId = "test-model-id";
        UpdateBlockedModelRequest updateBlockedModelRequest = new UpdateBlockedModelRequest(modelId, false);

        assertEquals(modelId, updateBlockedModelRequest.getModelId());
    }

    public void testIsRemoveRequest() {
        String modelId = "test-model-id";
        boolean isRemoveRequest = false;
        UpdateBlockedModelRequest updateBlockedModelRequest = new UpdateBlockedModelRequest(modelId, isRemoveRequest);

        assertEquals(isRemoveRequest, updateBlockedModelRequest.isRemoveRequest());
    }
}
