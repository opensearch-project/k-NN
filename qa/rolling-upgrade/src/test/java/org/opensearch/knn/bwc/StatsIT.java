/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.knn.plugin.stats.KNNStats;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;

public class StatsIT extends AbstractRollingUpgradeTestCase {
    private KNNStats knnStats;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        this.knnStats = spy(new KNNStats());
    }

    // Validate if all the KNN Stats metrics from old version are present in new version
    public void testAllMetricStatsReturned() throws Exception {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> clusterStats = parseClusterStatsResponse(responseBody);
        assertNotNull(clusterStats);
        assertTrue(knnStats.getClusterStats().keySet().containsAll(clusterStats.keySet()));
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(responseBody);
        assertNotNull(nodeStats.get(0));
        doReturn(randomBoolean()).when(knnStats).isRemoteBuildEnabled();
        assertTrue(knnStats.getNodeStats().keySet().containsAll(nodeStats.get(0).keySet()));
    }

    // Verify if it returns failure for invalid metric
    public void testInvalidMetricsStats() {
        expectThrows(ResponseException.class, () -> getKnnStats(Collections.emptyList(), Collections.singletonList("invalid_metric")));
    }

}
