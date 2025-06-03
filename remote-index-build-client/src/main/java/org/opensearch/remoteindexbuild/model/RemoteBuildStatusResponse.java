/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.Builder;
import lombok.Value;
import org.opensearch.core.ParseField;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;

import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.ERROR_MESSAGE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.FILE_NAME;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.TASK_STATUS;

/**
 * Response from the remote index build service. This class is used to parse the response from the remote index build service.
 */
@Value
@Builder
public class RemoteBuildStatusResponse {
    private static final ParseField TASK_STATUS_FIELD = new ParseField(TASK_STATUS);
    private static final ParseField FILE_NAME_FIELD = new ParseField(FILE_NAME);
    private static final ParseField ERROR_MESSAGE_FIELD = new ParseField(ERROR_MESSAGE);

    String taskStatus;
    String fileName;
    String errorMessage;

    /**
     * Parse the response from the remote index build service status API.
     * <p>
     * Example response to be parsed:
     * <pre>{@code {
     *     "task_status" : "String", // one of RUNNING_INDEX_BUILD, FAILED_INDEX_BUILD, COMPLETED_INDEX_BUILD
     *     "file_name" : "String"
     *     "error_message": "String"
     * } }</pre>
     */
    public static RemoteBuildStatusResponse fromXContent(XContentParser parser) throws IOException {
        final RemoteBuildStatusResponseBuilder builder = new RemoteBuildStatusResponseBuilder();
        XContentParser.Token token = parser.nextToken();
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IOException("Invalid response format, was expecting a " + XContentParser.Token.START_OBJECT);
        }
        String currentFieldName = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (TASK_STATUS_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.taskStatus(parser.text());
                } else if (FILE_NAME_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.fileName(parser.text());
                } else if (ERROR_MESSAGE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.errorMessage(parser.text());
                } else {
                    throw new IOException("Invalid response format, unknown field: " + currentFieldName);
                }
            }
        }
        return builder.build();
    }
}
