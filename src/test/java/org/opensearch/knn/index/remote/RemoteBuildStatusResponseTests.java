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

public class RemoteBuildStatusResponseTests extends KNNTestCase {
    public static final String UNKNOWN_RESPONSE = "{"
        + "\"task_status\":\"RUNNING_INDEX_BUILD\","
        + "\"error_message\":null,"
        + "\"unknown_field\":\"value\""
        + "}";
    public static final String COMPLETED_RESPONSE = "{"
        + "\"task_status\":\"COMPLETED_INDEX_BUILD\","
        + "\"index_path\":\"/path/to/index\","
        + "\"error_message\":null"
        + "}";
    public static final String MISSING_TASK_STATUS_RESPONSE = "{" + "\"index_path\":\"/path/to/index\"," + "\"error_message\":null" + "}";
    public static final String MISSING_INDEX_PATH_RESPONSE = "{"
        + "\"task_status\":\"COMPLETED_INDEX_BUILD\","
        + "\"error_message\":null"
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
            assertEquals("COMPLETED_INDEX_BUILD", response.getTaskStatus());
            assertEquals("/path/to/index", response.getIndexPath());
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
            assertEquals("Invalid response format, missing task_status", exception.getMessage());
        }
    }

    public void testMissingIndexPathForCompletedStatus() throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                MISSING_INDEX_PATH_RESPONSE
            )
        ) {
            IOException exception = assertThrows(IOException.class, () -> RemoteBuildStatusResponse.fromXContent(parser));
            assertEquals("Invalid response format, missing index_path for completed status", exception.getMessage());
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
            assertEquals("Invalid response format, unknown field: unknown_field", exception.getMessage());
        }
    }
}
