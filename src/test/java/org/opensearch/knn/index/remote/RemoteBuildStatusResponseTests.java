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

import static org.opensearch.knn.index.remote.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ERROR_MESSAGE;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FILE_NAME;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.RUNNING_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.TASK_STATUS;

public class RemoteBuildStatusResponseTests extends KNNTestCase {
    public static final String UNKNOWN_FIELD = "unknown_field";
    public static final String UNKNOWN_VALUE = "value";
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
    public static final String MISSING_TASK_STATUS_RESPONSE = "{"
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
    public static final String MISSING_FILE_NAME_RESPONSE = "{"
        + "\""
        + TASK_STATUS
        + "\":\""
        + COMPLETED_INDEX_BUILD
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
            assertNotNull(response);
            assertEquals(COMPLETED_INDEX_BUILD, response.getTaskStatus());
            assertEquals(MOCK_FILE_NAME, response.getFileName());
            assertNull(response.getErrorMessage());
        }
    }

    public void testMissingTaskStatus() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                MISSING_TASK_STATUS_RESPONSE
            )
        ) {
            IOException exception = assertThrows(IOException.class, () -> RemoteBuildStatusResponse.fromXContent(parser));
            assertEquals("Invalid response format, missing " + TASK_STATUS, exception.getMessage());
        }
    }

    public void testMissingIndexPathForCompletedStatus() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                MISSING_FILE_NAME_RESPONSE
            )
        ) {
            IOException exception = assertThrows(IOException.class, () -> RemoteBuildStatusResponse.fromXContent(parser));
            assertEquals("Invalid response format, missing " + FILE_NAME + " for completed status", exception.getMessage());
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
            IOException exception = assertThrows(IOException.class, () -> RemoteBuildStatusResponse.fromXContent(parser));
            assertEquals("Invalid response format, unknown field: " + UNKNOWN_FIELD, exception.getMessage());
        }
    }
}
