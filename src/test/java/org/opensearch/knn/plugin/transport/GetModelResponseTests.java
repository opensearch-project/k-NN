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

import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNClusterUtil;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class GetModelResponseTests extends KNNTestCase {

    private ModelMetadata getModelMetadata(ModelState state) {
        return new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.DEFAULT,
            4,
            state,
            "2021-03-27 10:15:30 AM +05:30",
            "test model",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT
        );
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
        try (MockedStatic<KNNClusterUtil> knnClusterUtilMockedStatic = mockStatic(KNNClusterUtil.class)) {
            final KNNClusterUtil knnClusterUtil = mock(KNNClusterUtil.class);
            when(knnClusterUtil.getClusterMinVersion()).thenReturn(Version.CURRENT);
            knnClusterUtilMockedStatic.when(KNNClusterUtil::instance).thenReturn(knnClusterUtil);
            String modelId = "test-model";
            byte[] testModelBlob = "hello".getBytes();
            Model model = new Model(getModelMetadata(ModelState.CREATED), testModelBlob, modelId);
            GetModelResponse getModelResponse = new GetModelResponse(model);
            String expectedResponseString =
                "{\"model_id\":\"test-model\",\"model_blob\":\"aGVsbG8=\",\"state\":\"created\",\"timestamp\":\"2021-03-27 10:15:30 AM +05:30\",\"description\":\"test model\",\"error\":\"\",\"space_type\":\"l2\",\"dimension\":4,\"engine\":\"nmslib\",\"training_node_assignment\":\"\",\"model_definition\":{\"name\":\"\",\"parameters\":{}},\"data_type\":\"float\"}";
            XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
            getModelResponse.toXContent(xContentBuilder, null);
            assertEquals(expectedResponseString, xContentBuilder.toString());
        }
    }

    public void testXContentWithNoModelBlob() throws IOException {
        try (MockedStatic<KNNClusterUtil> knnClusterUtilMockedStatic = mockStatic(KNNClusterUtil.class)) {
            final KNNClusterUtil knnClusterUtil = mock(KNNClusterUtil.class);
            when(knnClusterUtil.getClusterMinVersion()).thenReturn(Version.CURRENT);
            knnClusterUtilMockedStatic.when(KNNClusterUtil::instance).thenReturn(knnClusterUtil);
            String modelId = "test-model";
            Model model = new Model(getModelMetadata(ModelState.FAILED), null, modelId);
            GetModelResponse getModelResponse = new GetModelResponse(model);
            String expectedResponseString =
                "{\"model_id\":\"test-model\",\"model_blob\":\"\",\"state\":\"failed\",\"timestamp\":\"2021-03-27 10:15:30 AM +05:30\",\"description\":\"test model\",\"error\":\"\",\"space_type\":\"l2\",\"dimension\":4,\"engine\":\"nmslib\",\"training_node_assignment\":\"\",\"model_definition\":{\"name\":\"\",\"parameters\":{}},\"data_type\":\"float\"}";
            XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
            getModelResponse.toXContent(xContentBuilder, null);
            assertEquals(expectedResponseString, xContentBuilder.toString());
        }
    }
}
