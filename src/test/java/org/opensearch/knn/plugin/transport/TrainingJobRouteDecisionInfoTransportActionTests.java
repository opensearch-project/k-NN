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

import org.mockito.MockedStatic;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.training.TrainingJobRunner;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.util.Collections;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class TrainingJobRouteDecisionInfoTransportActionTests extends KNNTestCase {
    public void testNodeOperation() {
        // Initialize mocked variables for the class
        DiscoveryNode node = mock(DiscoveryNode.class);
        when(clusterService.localNode()).thenReturn(node);
        ThreadPool threadPool = mock(ThreadPool.class);
        TransportService transportService = mock(TransportService.class);
        doNothing().when(transportService).registerRequestHandler(any(), any(), any(), any());
        ActionFilters actionFilters = new ActionFilters(Collections.emptySet());

        TrainingJobRouteDecisionInfoTransportAction trainingJobRouteDecisionInfoTransportAction =
            new TrainingJobRouteDecisionInfoTransportAction(threadPool, clusterService, transportService, actionFilters);

        try (MockedStatic<TrainingJobRunner> mockedTrainingJobRunnerStatic = mockStatic(TrainingJobRunner.class)) {
            // Ensure the job count is correct
            int initialJobCount = 4;
            final TrainingJobRunner mockedTrainingJobRunner = mock(TrainingJobRunner.class);
            when(mockedTrainingJobRunner.getJobCount()).thenReturn(initialJobCount);
            mockedTrainingJobRunnerStatic.when(TrainingJobRunner::getInstance).thenReturn(mockedTrainingJobRunner);

            TrainingJobRouteDecisionInfoNodeRequest request = new TrainingJobRouteDecisionInfoNodeRequest();
            TrainingJobRouteDecisionInfoNodeResponse response = trainingJobRouteDecisionInfoTransportAction.nodeOperation(request);
            assertEquals(initialJobCount, response.getTrainingJobCount().intValue());

            int resetJobCount = 0;
            when(mockedTrainingJobRunner.getJobCount()).thenReturn(resetJobCount);
            response = trainingJobRouteDecisionInfoTransportAction.nodeOperation(request);
            assertEquals(resetJobCount, response.getTrainingJobCount().intValue());
        }
    }
}
