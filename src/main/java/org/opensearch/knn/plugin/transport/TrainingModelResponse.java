/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionResponse;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;

import java.io.IOException;

public class TrainingModelResponse extends ActionResponse implements ToXContentObject {

    public TrainingModelResponse(StreamInput streamInput) {
    }

    @Override
    public void writeTo(StreamOutput streamOutput) throws IOException {

    }

    @Override
    public XContentBuilder toXContent(XContentBuilder xContentBuilder, Params params) throws IOException {
        return null;
    }
}
