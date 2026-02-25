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

import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.JOB_ID_FIELD;

/**
 * Response from the remote index build service. This class is used to parse the response from the remote index build service.
 */
@Value
@Builder
public class RemoteBuildResponse {
    private static final ParseField JOB_ID = new ParseField(JOB_ID_FIELD);
    String jobId;

    /**
     * Parse the response from the remote index build service build API.
     * <p>
     * Example response:
     *<pre>{@code {
     *  "job_id" : "String"
     * } }</pre>
     */
    public static RemoteBuildResponse fromXContent(XContentParser parser) throws IOException {
        final RemoteBuildResponseBuilder builder = new RemoteBuildResponseBuilder();
        XContentParser.Token token = parser.nextToken();
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IOException("Invalid response format, was expecting a " + XContentParser.Token.START_OBJECT);
        }
        String currentFieldName = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (JOB_ID.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.jobId(parser.text());
                } else {
                    throw new IOException("Invalid response format, unknown field: " + currentFieldName);
                }
            }
        }
        return builder.build();
    }
}
