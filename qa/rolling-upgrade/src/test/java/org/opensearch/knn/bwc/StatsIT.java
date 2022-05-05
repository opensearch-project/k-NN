/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.cluster.health.ClusterHealthStatus;
import org.opensearch.common.xcontent.XContentType;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.plugin.stats.StatNames.MODEL_INDEX_STATUS;

public class StatsIT extends AbstractRollingUpgradeTestCase {
    private static final String MODEL_INDEX_STATUS_NAME = MODEL_INDEX_STATUS.getName();

    // KNN Stats : model_index_status
    public void testModelIndexHealthMetrics() throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                Map<String, Object> statsMap = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                assertNull(statsMap.get(MODEL_INDEX_STATUS_NAME));

                createModelSystemIndex();
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    Map<String, Object> statsMapFirstMixedRound = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                    assertNotNull(statsMapFirstMixedRound.get(MODEL_INDEX_STATUS_NAME));
                    assertNotNull(ClusterHealthStatus.fromString((String) statsMapFirstMixedRound.get(MODEL_INDEX_STATUS_NAME)));
                } else {
                    deleteKNNIndex(MODEL_INDEX_NAME);

                    Map<String, Object> statsMapSecondMixedRound = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                    assertNull(statsMapSecondMixedRound.get(MODEL_INDEX_STATUS_NAME));

                    createModelSystemIndex();
                }
                break;
            case UPGRADED:
                Map<String, Object> statsMapUpgraded = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                assertNotNull(statsMapUpgraded.get(MODEL_INDEX_STATUS_NAME));
                assertNotNull(ClusterHealthStatus.fromString((String) statsMapUpgraded.get(MODEL_INDEX_STATUS_NAME)));

                deleteKNNIndex(MODEL_INDEX_NAME);
        }
    }

    public Map<String, Object> getModelIndexHealthMetric(String modelIndexStatusName) throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(modelIndexStatusName));
        String responseBody = EntityUtils.toString(response.getEntity());
        return createParser(XContentType.JSON.xContent(), responseBody).map();
    }

}
