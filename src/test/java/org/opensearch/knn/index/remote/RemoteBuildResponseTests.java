/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.MOCK_JOB_ID;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.MOCK_JOB_ID_RESPONSE;

public class RemoteBuildResponseTests extends KNNTestCase {
    public void testRemoteBuildResponseParsing() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                MOCK_JOB_ID_RESPONSE
            )
        ) {
            RemoteBuildResponse response = RemoteBuildResponse.fromXContent(parser);
            assertNotNull(response);
            assertEquals(MOCK_JOB_ID, response.getJobId());
        }
    }

    public void testRemoteBuildResponseParsingError() throws IOException {
        String jsonResponse = "{\"error\":\"test-error\"}";
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                jsonResponse
            )
        ) {
            assertThrows(IOException.class, () -> RemoteBuildResponse.fromXContent(parser));
        }
    }
}
