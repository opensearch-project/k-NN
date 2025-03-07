/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.remoteindexbuild.TestConstants;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;

public class RemoteBuildResponseTests extends OpenSearchTestCase {
    public void testRemoteBuildResponseParsing() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                TestConstants.MOCK_JOB_ID_RESPONSE
            )
        ) {
            RemoteBuildResponse response = RemoteBuildResponse.fromXContent(parser);
            assertNotNull(response);
            assertEquals(TestConstants.MOCK_JOB_ID, response.getJobId());
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
