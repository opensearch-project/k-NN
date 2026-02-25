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

import com.google.common.collect.Maps;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.Map;

public class TrainingModelResponseTests extends KNNTestCase {

    public void testStreams() throws IOException {
        String modelId = "test-model-id";

        TrainingModelResponse original = new TrainingModelResponse(modelId);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        original.writeTo(streamOutput);

        TrainingModelResponse copy = new TrainingModelResponse(streamOutput.bytes().streamInput());

        assertEquals(original.getModelId(), copy.getModelId());
    }

    public void testGetModelId() {
        String modelId = "test-model-id";

        TrainingModelResponse trainingModelResponse = new TrainingModelResponse(modelId);

        assertEquals(modelId, trainingModelResponse.getModelId());
    }

    public void testToXContent() throws IOException {
        String modelId = "test-model-id";

        TrainingModelResponse response = new TrainingModelResponse(modelId);

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder = response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> actual = xContentBuilderToMap(builder);

        // We expect this:
        // {
        // "model_id": "test-model-id"
        // }
        XContentBuilder expectedXContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(KNNConstants.MODEL_ID, modelId)
            .endObject();
        Map<String, Object> expected = xContentBuilderToMap(expectedXContentBuilder);

        // Check responses are equal
        assertTrue(Maps.difference(expected, actual).areEqual());
    }

}
