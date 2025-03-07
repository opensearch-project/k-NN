/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

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

@SuppressWarnings("unchecked")
public class RemoteFaissHNSWIndexParametersTests extends OpenSearchTestCase {
    public void testToXContent() throws IOException {
        RemoteFaissHNSWIndexParameters params = spy(
            RemoteFaissHNSWIndexParameters.builder()
                .spaceType(TestConstants.L2_SPACE_TYPE)
                .algorithm(TestConstants.HNSW_ALGORITHM)
                .m(16)
                .efConstruction(88)
                .efSearch(99)
                .build()
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
            assertEquals(16, algorithmParams.get(KNNRemoteConstants.METHOD_PARAMETER_M));
            assertEquals(88, algorithmParams.get(KNNRemoteConstants.METHOD_PARAMETER_EF_CONSTRUCTION));
            assertEquals(99, algorithmParams.get(KNNRemoteConstants.METHOD_PARAMETER_EF_SEARCH));
        }

        verify(params).addAlgorithmParameters(any(XContentBuilder.class));
    }
}
