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

import org.opensearch.common.Strings;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeParserRegistry;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;

public class GetModelResponseTests extends KNNTestCase {

    private ModelMetadata getModelMetadata(ModelState state) {
        return new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 4, state, "2021-03-27 10:15:30 AM +05:30", "test model", "");
    }

    public void testStreams() throws IOException {
        String modelId = "test-model";
        byte[] testModelBlob = "hello".getBytes();
        Model model = new Model(getModelMetadata(ModelState.CREATED), testModelBlob, modelId);
        GetModelResponse getModelResponse = new GetModelResponse(model);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        getModelResponse.writeTo(streamOutput);
        GetModelResponse getModelResponseCopy = new GetModelResponse(streamOutput.bytes().streamInput());
        assertEquals(getModelResponse.getModel(), getModelResponseCopy.getModel());
    }

    public void testXContent() throws IOException {
        String modelId = "test-model";
        byte[] testModelBlob = "hello".getBytes();
        Model model = new Model(getModelMetadata(ModelState.CREATED), testModelBlob, modelId);
        GetModelResponse getModelResponse = new GetModelResponse(model);
        String expectedResponseString =
            "{\"model_id\":\"test-model\",\"model_blob\":\"aGVsbG8=\",\"state\":\"created\",\"timestamp\":\"2021-03-27 10:15:30 AM +05:30\",\"description\":\"test model\",\"error\":\"\",\"space_type\":\"l2\",\"dimension\":4,\"engine\":\"nmslib\"}";
        XContentBuilder xContentBuilder = XContentFactory.contentBuilder(MediaTypeParserRegistry.getDefaultMediaType());
        getModelResponse.toXContent(xContentBuilder, null);
        assertEquals(expectedResponseString, Strings.toString(xContentBuilder));
    }

    public void testXContentWithNoModelBlob() throws IOException {
        String modelId = "test-model";
        Model model = new Model(getModelMetadata(ModelState.FAILED), null, modelId);
        GetModelResponse getModelResponse = new GetModelResponse(model);
        String expectedResponseString =
            "{\"model_id\":\"test-model\",\"model_blob\":\"\",\"state\":\"failed\",\"timestamp\":\"2021-03-27 10:15:30 AM +05:30\",\"description\":\"test model\",\"error\":\"\",\"space_type\":\"l2\",\"dimension\":4,\"engine\":\"nmslib\"}";
        XContentBuilder xContentBuilder = XContentFactory.contentBuilder(MediaTypeParserRegistry.getDefaultMediaType());
        getModelResponse.toXContent(xContentBuilder, null);
        assertEquals(expectedResponseString, Strings.toString(xContentBuilder));
    }
}
