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
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;

public class UpdateModelMetadataRequestTests extends KNNTestCase {

    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        String modelId = "test-model";
        boolean isRemoveRequest = false;

        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT
        );
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest(modelId, isRemoveRequest, modelMetadata);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        updateModelMetadataRequest.writeTo(streamOutput);

        UpdateModelMetadataRequest updateModelMetadataRequestCopy = new UpdateModelMetadataRequest(streamOutput.bytes().streamInput());

        assertEquals(updateModelMetadataRequest.getModelId(), updateModelMetadataRequestCopy.getModelId());
        assertEquals(updateModelMetadataRequest.isRemoveRequest(), updateModelMetadataRequestCopy.isRemoveRequest());
        assertEquals(updateModelMetadataRequest.getModelMetadata(), updateModelMetadataRequestCopy.getModelMetadata());
    }

    public void testValidate() {

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

        UpdateModelMetadataRequest updateModelMetadataRequest1 = new UpdateModelMetadataRequest("test", true, null);
        assertNull(updateModelMetadataRequest1.validate());

        UpdateModelMetadataRequest updateModelMetadataRequest2 = new UpdateModelMetadataRequest("test", false, null);
        assertNotNull(updateModelMetadataRequest2.validate());

        UpdateModelMetadataRequest updateModelMetadataRequest3 = new UpdateModelMetadataRequest("test", false, modelMetadata);
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
        UpdateModelMetadataRequest updateModelMetadataRequest = new UpdateModelMetadataRequest("test", true, modelMetadata);

        assertEquals(modelMetadata, updateModelMetadataRequest.getModelMetadata());
    }
}
