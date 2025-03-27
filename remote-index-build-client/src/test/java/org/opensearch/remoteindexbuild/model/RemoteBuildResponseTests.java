/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import org.junit.Assert;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;

import static org.opensearch.remoteindexbuild.TestConstants.UNKNOWN_FIELD;
import static org.opensearch.remoteindexbuild.TestConstants.UNKNOWN_VALUE;

public class RemoteBuildResponseTests extends OpenSearchTestCase {
    public static final String MOCK_JOB_ID_RESPONSE = "{\"job_id\": \"job-1739930402\"}";
    public static final String MOCK_JOB_ID = "job-1739930402";

    public void testRemoteBuildResponseParsing() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                MOCK_JOB_ID_RESPONSE
            )
        ) {
            RemoteBuildResponse response = RemoteBuildResponse.fromXContent(parser);
            Assert.assertNotNull(response);
            Assert.assertEquals(MOCK_JOB_ID, response.getJobId());
        }
    }

    public void testRemoteBuildResponseParsingError() throws IOException {
        String jsonResponse = "{\"" + UNKNOWN_FIELD + "\":\"" + UNKNOWN_VALUE + "\"}";
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                jsonResponse
            )
        ) {
            Assert.assertThrows(IOException.class, () -> RemoteBuildResponse.fromXContent(parser));
        }
    }
}
