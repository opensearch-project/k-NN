/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.experimental.SuperBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.remoteindexbuild.TestConstants;
import org.opensearch.remoteindexbuild.constants.KNNRemoteConstants;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.opensearch.core.xcontent.DeprecationHandler.THROW_UNSUPPORTED_OPERATION;

public class RemoteIndexParametersTests extends OpenSearchTestCase {
    @SuperBuilder
    private static class TestRemoteIndexParameters extends RemoteIndexParameters {

        public static final String TEST_PARAM = "test_param";
        public static final String TEST_VALUE = "test_value";

        protected TestRemoteIndexParameters(RemoteIndexParametersBuilder<?, ?> b) {
            super(b);
        }

        @Override
        void addAlgorithmParameters(XContentBuilder builder) throws IOException {
            builder.startObject(KNNRemoteConstants.ALGORITHM_PARAMETERS);
            builder.field(TEST_PARAM, TEST_VALUE);
            builder.endObject();
        }
    }

    @SuppressWarnings("unchecked")
    public void testToXContent() throws IOException {
        TestRemoteIndexParameters params = spy(
            TestRemoteIndexParameters.builder().spaceType(TestConstants.L2_SPACE_TYPE).algorithm(TestConstants.HNSW_ALGORITHM).build()
        );

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

            assertEquals(TestConstants.L2_SPACE_TYPE, map.get(KNNRemoteConstants.METHOD_PARAMETER_SPACE_TYPE));
            assertEquals(TestConstants.HNSW_ALGORITHM, map.get(KNNRemoteConstants.ALGORITHM));

            Map<String, Object> algorithmParams = (Map<String, Object>) map.get(KNNRemoteConstants.ALGORITHM_PARAMETERS);
            assertEquals(TestRemoteIndexParameters.TEST_VALUE, algorithmParams.get(TestRemoteIndexParameters.TEST_PARAM));
        }
        verify(params).addAlgorithmParameters(any(XContentBuilder.class));
    }
}
