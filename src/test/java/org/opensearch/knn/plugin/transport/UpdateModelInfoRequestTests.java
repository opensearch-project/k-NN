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

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelInfo;

import java.io.IOException;

public class UpdateModelInfoRequestTests extends KNNTestCase {

    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        String modelId = "test-model";
        boolean isRemoveRequest = false;

        ModelInfo modelInfo = new ModelInfo(knnEngine, spaceType, dimension);
        UpdateModelInfoRequest updateModelInfoRequest = new UpdateModelInfoRequest(modelId, isRemoveRequest, modelInfo);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        updateModelInfoRequest.writeTo(streamOutput);

        UpdateModelInfoRequest updateModelInfoRequestCopy = new UpdateModelInfoRequest(streamOutput.bytes().streamInput());

        assertEquals(updateModelInfoRequest.getModelId(), updateModelInfoRequestCopy.getModelId());
        assertEquals(updateModelInfoRequest.isRemoveRequest(), updateModelInfoRequestCopy.isRemoveRequest());
        assertEquals(updateModelInfoRequest.getModelInfo(), updateModelInfoRequestCopy.getModelInfo());
    }

    public void testValidate() {
        ModelInfo modelInfo = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);

        UpdateModelInfoRequest updateModelInfoRequest1 = new UpdateModelInfoRequest("test", true, null);
        assertNull(updateModelInfoRequest1.validate());

        UpdateModelInfoRequest updateModelInfoRequest2 = new UpdateModelInfoRequest("test", false, null);
        assertNotNull(updateModelInfoRequest2.validate());

        UpdateModelInfoRequest updateModelInfoRequest3 = new UpdateModelInfoRequest("test", false, modelInfo);
        assertNull(updateModelInfoRequest3.validate());

        UpdateModelInfoRequest updateModelInfoRequest4 = new UpdateModelInfoRequest("", true, null);
        assertNotNull(updateModelInfoRequest4.validate());
    }

    public void testGetModelId() {
        String modelId = "test-model";
        UpdateModelInfoRequest updateModelInfoRequest = new UpdateModelInfoRequest(modelId, true, null);

        assertEquals(modelId, updateModelInfoRequest.getModelId());
    }

    public void testIsRemoveRequest() {
        boolean isRemoveRequest = false;
        UpdateModelInfoRequest updateModelInfoRequest = new UpdateModelInfoRequest("test", isRemoveRequest, null);

        assertEquals(isRemoveRequest, updateModelInfoRequest.isRemoveRequest());
    }

    public void testGetModelMetadata() {
        ModelInfo modelInfo = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);
        UpdateModelInfoRequest updateModelInfoRequest = new UpdateModelInfoRequest("test", true, modelInfo);

        assertEquals(modelInfo, updateModelInfoRequest.getModelInfo());
    }
}
