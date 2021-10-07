package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.nodes.BaseNodesResponse;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;
import java.util.List;

public class RemoveModelFromCacheResponse extends BaseNodesResponse<TrainingJobRouteDecisionInfoNodeResponse> {

    protected RemoveModelFromCacheResponse(StreamInput in) throws IOException {
        super(in);
    }

    @Override
    protected List<TrainingJobRouteDecisionInfoNodeResponse> readNodesFrom(StreamInput streamInput) throws IOException {
        return null;
    }

    @Override
    protected void writeNodesTo(StreamOutput streamOutput, List<TrainingJobRouteDecisionInfoNodeResponse> list) throws IOException {

    }
}
