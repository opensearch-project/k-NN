/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.index.shard.IndexShardTestUtils;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;

import java.util.List;
import java.util.Map;

public class KNNNodeStatAggregationTests extends KNNTestCase {

    DiscoveryNode fakeNode = IndexShardTestUtils.getFakeDiscoNode("test");

    public void testParse() {
        List<KNNStatsNodeResponse> responseList = List.of(
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true))
        );
        assertTrue(new KNNNodeStatAggregation(responseList).isClusterLevelCircuitBreakerTripped());

        responseList = List.of(
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false))
        );
        assertFalse(new KNNNodeStatAggregation(responseList).isClusterLevelCircuitBreakerTripped());

        responseList = List.of(
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), true)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false)),
            new KNNStatsNodeResponse(fakeNode, Map.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false))
        );
        assertTrue(new KNNNodeStatAggregation(responseList).isClusterLevelCircuitBreakerTripped());
    }

}
