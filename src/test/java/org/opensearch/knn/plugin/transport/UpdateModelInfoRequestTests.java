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
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest(modelId, isRemoveRequest, modelInfo);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        updateModelMetadataRequest.writeTo(streamOutput);

        UpdateModelMetadataRequest updateModelMetadataRequestCopy = new UpdateModelMetadataRequest(streamOutput.bytes().streamInput());

        assertEquals(updateModelMetadataRequest.getModelId(), updateModelMetadataRequestCopy.getModelId());
        assertEquals(updateModelMetadataRequest.isRemoveRequest(), updateModelMetadataRequestCopy.isRemoveRequest());
        assertEquals(updateModelMetadataRequest.getModelInfo(), updateModelMetadataRequestCopy.getModelInfo());
    }

    public void testValidate() {
        ModelInfo modelInfo = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);

        UpdateModelMetadataRequest updateModelMetadataRequest1 = new UpdateModelMetadataRequest("test", true, null);
        assertNull(updateModelMetadataRequest1.validate());

        UpdateModelMetadataRequest updateModelMetadataRequest2 = new UpdateModelMetadataRequest("test", false, null);
        assertNotNull(updateModelMetadataRequest2.validate());

        UpdateModelMetadataRequest updateModelMetadataRequest3 = new UpdateModelMetadataRequest("test", false, modelInfo);
        assertNull(updateModelMetadataRequest3.validate());

        UpdateModelMetadataRequest updateModelMetadataRequest4 = new UpdateModelMetadataRequest("", true, null);
        assertNotNull(updateModelMetadataRequest4.validate());
    }

    public void testGetModelId() {
        String modelId = "test-model";
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest(modelId, true, null);

        assertEquals(modelId, updateModelMetadataRequest.getModelId());
    }

    public void testIsRemoveRequest() {
        boolean isRemoveRequest = false;
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest("test", isRemoveRequest, null);

        assertEquals(isRemoveRequest, updateModelMetadataRequest.isRemoveRequest());
    }

    public void testGetModelMetadata() {
        ModelInfo modelInfo = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest("test", true, modelInfo);

        assertEquals(modelInfo, updateModelMetadataRequest.getModelInfo());
    }
}
