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
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.ERROR_MESSAGE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.FILE_NAME;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.RUNNING_INDEX_BUILD;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.TASK_STATUS;

public class RemoteBuildStatusResponseTests extends OpenSearchTestCase {
    public static final String NULL = "null";
    public static final String MOCK_FILE_NAME = "graph.faiss";
    public static final String UNKNOWN_RESPONSE = "{"
        + "\""
        + TASK_STATUS
        + "\":\""
        + RUNNING_INDEX_BUILD
        + "\","
        + "\""
        + ERROR_MESSAGE
        + "\":"
        + NULL
        + ","
        + "\""
        + UNKNOWN_FIELD
        + "\":\""
        + UNKNOWN_VALUE
        + "\""
        + "}";
    public static final String COMPLETED_RESPONSE = "{"
        + "\""
        + TASK_STATUS
        + "\":\""
        + COMPLETED_INDEX_BUILD
        + "\","
        + "\""
        + FILE_NAME
        + "\":\""
        + MOCK_FILE_NAME
        + "\","
        + "\""
        + ERROR_MESSAGE
        + "\":"
        + NULL
        + "}";

    public void testSuccessfulBuildStatusResponse() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                COMPLETED_RESPONSE
            )
        ) {
            RemoteBuildStatusResponse response = RemoteBuildStatusResponse.fromXContent(parser);
            Assert.assertNotNull(response);
            Assert.assertEquals(COMPLETED_INDEX_BUILD, response.getTaskStatus());
            Assert.assertEquals(MOCK_FILE_NAME, response.getFileName());
            Assert.assertNull(response.getErrorMessage());
        }
    }

    public void testUnknownField() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                UNKNOWN_RESPONSE
            )
        ) {
            IOException exception = Assert.assertThrows(IOException.class, () -> RemoteBuildStatusResponse.fromXContent(parser));
            Assert.assertEquals("Invalid response format, unknown field: " + UNKNOWN_FIELD, exception.getMessage());
        }
    }
}
