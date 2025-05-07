/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.AllArgsConstructor;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.knn.profiler.SegmentProfilerState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@AllArgsConstructor
public class KNNIndexShardProfileResult implements Writeable {
    List<SegmentProfilerState> segmentProfilerStateList;
    String shardId;

    /**
     * Constructor to deserialize from StreamInput
     *
     * @param in StreamInput to read from
     * @throws IOException if there's an error reading from the stream
     */
    public KNNIndexShardProfileResult(StreamInput in) throws IOException {
        this.shardId = in.readString();

        // Read the number of SegmentProfilerStates
        int size = in.readInt();
        this.segmentProfilerStateList = new ArrayList<>(size);

        // Read each SegmentProfilerState
        for (int i = 0; i < size; i++) {
            byte[] stateBytes = in.readByteArray();
            this.segmentProfilerStateList.add(SegmentProfilerState.fromBytes(stateBytes));
        }
    }

    @Override
    public void writeTo(StreamOutput streamOutput) throws IOException {
        streamOutput.writeString(shardId);

        // Write the segment profiler state list size
        streamOutput.writeInt(segmentProfilerStateList.size());

        // Write each SegmentProfilerState as bytes
        for (SegmentProfilerState state : segmentProfilerStateList) {
            byte[] stateBytes = state.toByteArray();
            streamOutput.writeByteArray(stateBytes);
        }
    }
}
