/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.opensearch.core.xcontent.DeprecationHandler.THROW_UNSUPPORTED_OPERATION;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.index.SpaceType.L2;

@SuppressWarnings("unchecked")
public class RemoteFaissHNSWIndexParametersTests extends KNNTestCase {
    public void testToXContent() throws IOException {
        RemoteFaissHNSWIndexParameters params = spy(
            RemoteFaissHNSWIndexParameters.builder()
                .spaceType(L2.getValue())
                .algorithm(METHOD_HNSW)
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

            assertEquals(L2.getValue(), map.get(METHOD_PARAMETER_SPACE_TYPE));
            assertEquals(METHOD_HNSW, map.get(KNNRemoteConstants.ALGORITHM));

            Map<String, Object> algorithmParams = (Map<String, Object>) map.get(KNNRemoteConstants.ALGORITHM_PARAMETERS);
            assertEquals(16, algorithmParams.get(KNNConstants.METHOD_PARAMETER_M));
            assertEquals(88, algorithmParams.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION));
            assertEquals(99, algorithmParams.get(KNNConstants.METHOD_PARAMETER_EF_SEARCH));
        }

        verify(params).addAlgorithmParameters(any(XContentBuilder.class));
    }
}
