/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Builder;
import lombok.Value;
import org.apache.commons.lang.StringUtils;
import org.opensearch.core.ParseField;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;

import static org.opensearch.knn.index.remote.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ERROR_MESSAGE;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FILE_NAME;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.TASK_STATUS;

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

    static RemoteBuildStatusResponse fromXContent(XContentParser parser) throws IOException {
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
        if (StringUtils.isBlank(builder.taskStatus)) {
            throw new IOException("Invalid response format, missing " + TASK_STATUS);
        }
        if (COMPLETED_INDEX_BUILD.equals(builder.taskStatus) && StringUtils.isBlank(builder.fileName)) {
            throw new IOException("Invalid response format, missing " + FILE_NAME + " for completed status");
        }
        return builder.build();
    }
}
