/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.profiler.SegmentProfilerState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.closeTo;
import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;

public class KNNProfileTransportActionTests extends KNNTestCase {

    private static final int DIMENSION = 4;
    private static final double DELTA = 0.0001;

    private List<KNNIndexShardProfileResult> shardProfileResults;
    private KNNProfileResponse response;

    @Before
    public void setup() {
        shardProfileResults = new ArrayList<>();

        for (int shardId = 0; shardId < 2; shardId++) {
            List<SegmentProfilerState> segmentStates = new ArrayList<>();

            for (int segId = 0; segId < 2; segId++) {
                List<SummaryStatistics> stats = new ArrayList<>();

                for (int dim = 0; dim < DIMENSION; dim++) {
                    SummaryStatistics dimStats = new SummaryStatistics();
                    dimStats.addValue(1.0 + shardId + segId);
                    dimStats.addValue(2.0 + shardId + segId);
                    dimStats.addValue(3.0 + shardId + segId);
                    stats.add(dimStats);
                }

                segmentStates.add(new SegmentProfilerState(stats, DIMENSION, "_" + segId));
            }

            shardProfileResults.add(new KNNIndexShardProfileResult(segmentStates, String.valueOf(shardId)));
        }

        response = new KNNProfileResponse(shardProfileResults, 2, 2, 0, new ArrayList<>());
    }

    public void testShardAggregation() throws IOException {
        XContentBuilder builder = jsonBuilder();
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> responseMap = createParser(builder).map();

        Map<String, Object> shardProfiles = (Map<String, Object>) responseMap.get("shard_profiles");

        for (int shardId = 0; shardId < 2; shardId++) {
            Map<String, Object> shardProfile = (Map<String, Object>) shardProfiles.get(String.valueOf(shardId));
            Map<String, Object> aggregated = (Map<String, Object>) shardProfile.get("aggregated");
            List<Map<String, Object>> dimensions = (List<Map<String, Object>>) aggregated.get("dimensions");

            for (int dim = 0; dim < DIMENSION; dim++) {
                Map<String, Object> dimStats = dimensions.get(dim);

                double expectedMin = 1.0 + shardId;
                double expectedMax = 3.0 + shardId + 1;
                double expectedMean = (expectedMin + expectedMax) / 2.0;

                assertEquals(6, dimStats.get("count"));
                assertThat((Double) dimStats.get("min"), closeTo(expectedMin, DELTA));
                assertThat((Double) dimStats.get("max"), closeTo(expectedMax, DELTA));
                assertThat((Double) dimStats.get("mean"), closeTo(expectedMean, DELTA));
            }
        }
    }

    public void testClusterAggregation() throws IOException {
        XContentBuilder builder = jsonBuilder();
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> responseMap = createParser(builder).map();

        Map<String, Object> clusterAgg = (Map<String, Object>) responseMap.get("cluster_aggregation");
        List<Map<String, Object>> dimensions = (List<Map<String, Object>>) clusterAgg.get("dimensions");

        for (int dim = 0; dim < DIMENSION; dim++) {
            Map<String, Object> dimStats = dimensions.get(dim);
            assertEquals(12, dimStats.get("count"));
            assertThat((Double) dimStats.get("min"), closeTo(1.0, DELTA));
            assertThat((Double) dimStats.get("max"), closeTo(5.0, DELTA));
            assertThat((Double) dimStats.get("mean"), closeTo(3.0, DELTA));
        }
    }

    public void testFailedShards() throws IOException {
        List<DefaultShardOperationFailedException> failures = new ArrayList<>();
        failures.add(new DefaultShardOperationFailedException("test_index", 0, new RuntimeException("Test failure")));

        KNNProfileResponse responseWithFailures = new KNNProfileResponse(shardProfileResults, 2, 1, 1, failures);

        XContentBuilder builder = jsonBuilder();
        responseWithFailures.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> responseMap = createParser(builder).map();

        assertEquals(2, responseMap.get("total_shards"));
        assertEquals(1, responseMap.get("successful_shards"));
        assertEquals(1, responseMap.get("failed_shards"));

        List<Map<String, Object>> failuresList = (List<Map<String, Object>>) responseMap.get("failures");
        assertEquals(1, failuresList.size());
        assertEquals("test_index", failuresList.get(0).get("index"));
    }

    public void testEmptyResponse() throws IOException {
        KNNProfileResponse emptyResponse = new KNNProfileResponse(new ArrayList<>(), 0, 0, 0, new ArrayList<>());

        XContentBuilder builder = jsonBuilder();
        emptyResponse.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> responseMap = createParser(builder).map();

        assertEquals(0, responseMap.get("total_shards"));
        assertEquals(0, responseMap.get("successful_shards"));
        assertEquals(0, responseMap.get("failed_shards"));
        assertTrue(((Map<String, Object>) responseMap.get("shard_profiles")).isEmpty());
    }
}
