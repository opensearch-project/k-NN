/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import org.junit.Assert;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.remoteindexbuild.constants.KNNRemoteConstants;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.core.xcontent.DeprecationHandler.THROW_UNSUPPORTED_OPERATION;
import static org.opensearch.remoteindexbuild.TestConstants.HNSW_ALGORITHM;
import static org.opensearch.remoteindexbuild.TestConstants.L2_SPACE_TYPE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_M;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_SPACE_TYPE;

@SuppressWarnings("unchecked")
public class RemoteFaissHNSWIndexParametersTests extends OpenSearchTestCase {
    public void testToXContent() throws IOException {
        RemoteFaissHNSWIndexParameters params = Mockito.spy(
            RemoteFaissHNSWIndexParameters.builder()
                .spaceType(L2_SPACE_TYPE)
                .algorithm(HNSW_ALGORITHM)
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

            Assert.assertEquals(L2_SPACE_TYPE, map.get(METHOD_PARAMETER_SPACE_TYPE));
            Assert.assertEquals(HNSW_ALGORITHM, map.get(KNNRemoteConstants.ALGORITHM));

            Map<String, Object> algorithmParams = (Map<String, Object>) map.get(KNNRemoteConstants.ALGORITHM_PARAMETERS);
            Assert.assertEquals(16, algorithmParams.get(METHOD_PARAMETER_M));
            Assert.assertEquals(88, algorithmParams.get(METHOD_PARAMETER_EF_CONSTRUCTION));
            Assert.assertEquals(99, algorithmParams.get(METHOD_PARAMETER_EF_SEARCH));
        }

        Mockito.verify(params).addAlgorithmParameters(ArgumentMatchers.any(XContentBuilder.class));
    }
}
