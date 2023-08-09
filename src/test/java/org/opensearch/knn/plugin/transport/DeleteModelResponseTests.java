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
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

public class DeleteModelResponseTests extends KNNTestCase {

    public void testStreams() throws IOException {
        String modelId = "test-model";
        DeleteModelResponse deleteModelResponse = new DeleteModelResponse(modelId);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        deleteModelResponse.writeTo(streamOutput);
        DeleteModelResponse deleteModelResponseCopy = new DeleteModelResponse(streamOutput.bytes().streamInput());
        assertEquals(deleteModelResponse.getModelID(), deleteModelResponseCopy.getModelID());
        assertEquals(deleteModelResponse.getResult(), deleteModelResponseCopy.getResult());
        assertEquals(deleteModelResponse.getErrorMessage(), deleteModelResponseCopy.getErrorMessage());
    }

    public void testXContentWithoutError() throws IOException {
        String modelId = "test-model";
        DeleteModelResponse deleteModelResponse = new DeleteModelResponse(modelId);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        deleteModelResponse.writeTo(streamOutput);
        String expectedResponseString = "{\"model_id\":\"test-model\",\"result\":\"deleted\"}";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        deleteModelResponse.toXContent(xContentBuilder, null);
        assertEquals(expectedResponseString, xContentBuilder.toString());
    }
}