/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.experimental.SuperBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.opensearch.core.xcontent.DeprecationHandler.THROW_UNSUPPORTED_OPERATION;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.index.SpaceType.L2;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ALGORITHM;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ALGORITHM_PARAMETERS;

public class RemoteIndexParametersTests extends KNNTestCase {
    @SuperBuilder
    private static class TestRemoteIndexParameters extends RemoteIndexParameters {

        public static final String TEST_PARAM = "test_param";
        public static final String TEST_VALUE = "test_value";

        protected TestRemoteIndexParameters(RemoteIndexParametersBuilder<?, ?> b) {
            super(b);
        }

        @Override
        void addAlgorithmParameters(XContentBuilder builder) throws IOException {
            builder.startObject(ALGORITHM_PARAMETERS);
            builder.field(TEST_PARAM, TEST_VALUE);
            builder.endObject();
        }
    }

    @SuppressWarnings("unchecked")
    public void testToXContent() throws IOException {
        TestRemoteIndexParameters params = spy(TestRemoteIndexParameters.builder().spaceType(L2.getValue()).algorithm(METHOD_HNSW).build());

        XContentBuilder builder = XContentFactory.jsonBuilder();
        params.toXContent(builder, ToXContent.EMPTY_PARAMS);

        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                THROW_UNSUPPORTED_OPERATION,
                builder.toString()
            )
        ) {
            Map<String, Object> map = parser.map();

            assertEquals(L2.getValue(), map.get(METHOD_PARAMETER_SPACE_TYPE));
            assertEquals(METHOD_HNSW, map.get(ALGORITHM));

            Map<String, Object> algorithmParams = (Map<String, Object>) map.get(ALGORITHM_PARAMETERS);
            assertEquals(TestRemoteIndexParameters.TEST_VALUE, algorithmParams.get(TestRemoteIndexParameters.TEST_PARAM));
        }
        verify(params).addAlgorithmParameters(any(XContentBuilder.class));
    }
}
