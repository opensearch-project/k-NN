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

import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.ToXContentFragment;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.TRAINING_JOB_COUNT_FIELD_NAME;

/**
 * Node level response containing training job route decision info.
 */
public class TrainingJobRouteDecisionInfoNodeResponse extends BaseNodeResponse implements ToXContentFragment {

    private final Integer trainingJobCount;

    /**
     * Constructor
     *
     * @param in  stream
     * @throws IOException in case of I/O errors
     */
    public TrainingJobRouteDecisionInfoNodeResponse(StreamInput in) throws IOException {
        super(in);
        this.trainingJobCount = in.readInt();
    }

    /**
     * Constructor
     *
     * @param node node
     */
    public TrainingJobRouteDecisionInfoNodeResponse(DiscoveryNode node, Integer trainingJobCount) {
        super(node);
        this.trainingJobCount = trainingJobCount;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeInt(trainingJobCount);
    }

    /**
     * Getter for training job count
     *
     * @return The count of the training jobs on the node
     */
    public Integer getTrainingJobCount() {
        return trainingJobCount;
    }

    /**
     * Add training job route decision info to xcontent builder
     *
     * @param builder XContentBuilder
     * @param params Params
     * @return XContentBuilder
     * @throws IOException thrown by builder for invalid field
     */
    public XContentBuilder toXContent(XContentBuilder builder, ToXContent.Params params) throws IOException {
        builder.field(TRAINING_JOB_COUNT_FIELD_NAME, trainingJobCount);
        return builder;
    }
}
