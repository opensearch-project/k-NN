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

import com.google.common.collect.Maps;
import com.google.common.net.InetAddresses;
import org.opensearch.Version;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.transport.TransportAddress;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.net.InetAddress;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.TRAINING_JOB_COUNT_FIELD_NAME;

public class TrainingJobRouteDecisionInfoNodeResponseTests extends KNNTestCase {

    public void testStreams() throws IOException {
        BytesStreamOutput streamOutput = new BytesStreamOutput();

        int trainingJobCount = 13;
        InetAddress inetAddress = InetAddresses.fromInteger(randomInt());
        DiscoveryNode discoveryNode = new DiscoveryNode("id", new TransportAddress(inetAddress, 9200), Version.CURRENT);

        TrainingJobRouteDecisionInfoNodeResponse original = new TrainingJobRouteDecisionInfoNodeResponse(discoveryNode, trainingJobCount);

        original.writeTo(streamOutput);

        TrainingJobRouteDecisionInfoNodeResponse copy = new TrainingJobRouteDecisionInfoNodeResponse(streamOutput.bytes().streamInput());

        assertEquals(original.getTrainingJobCount(), copy.getTrainingJobCount());
    }

    public void testGetTrainingJobCount() {
        int trainingJobCount = 13;

        DiscoveryNode discoveryNode = mock(DiscoveryNode.class);

        TrainingJobRouteDecisionInfoNodeResponse response = new TrainingJobRouteDecisionInfoNodeResponse(discoveryNode, trainingJobCount);

        assertEquals(trainingJobCount, response.getTrainingJobCount().intValue());
    }

    public void testToXContent() throws IOException {
        int trainingJobCount = 13;

        // We expect this:
        // {
        // "training_job_count": 13
        // }
        XContentBuilder expectedXContentBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAINING_JOB_COUNT_FIELD_NAME, trainingJobCount)
                .endObject();

        Map<String, Object> expected = xContentBuilderToMap(expectedXContentBuilder);

        DiscoveryNode discoveryNode = mock(DiscoveryNode.class);
        TrainingJobRouteDecisionInfoNodeResponse response = new TrainingJobRouteDecisionInfoNodeResponse(discoveryNode, trainingJobCount);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder = response.toXContent(builder, ToXContent.EMPTY_PARAMS).endObject();
        Map<String, Object> actual = xContentBuilderToMap(builder);

        assertTrue(Maps.difference(expected, actual).areEqual());
    }
}