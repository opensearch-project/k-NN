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

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Assert;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * This IT class contains special cases of IT for segment replication behavior.
 * All indices created in this test have replication type SEGMENT, number of replicas: 1 and should be run on
 * at-least 2 node configuration.
 */
@Log4j2
public class SegmentReplicationIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "segment-replicated-knn-index";
    private static final int ASSERT_BUSY_TIMEOUT_SECONDS = 60;

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testSearchOnReplicas_whenIndexHasDeletedDocs_thenSuccess() {
        createKnnIndex(INDEX_NAME, getKNNSegmentReplicatedIndexSettings(), createKNNIndexMethodFieldMapping(FIELD_NAME, 2));
        ensureGreen(INDEX_NAME);

        final Float[] vector = { 1.3f, 2.2f };
        final int docsInIndex = 10;
        final int deleteDocs = 5;
        final int expectedVisibleDocs = docsInIndex - deleteDocs;

        for (int i = 0; i < docsInIndex; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
        }
        refreshIndex(INDEX_NAME);
        flushIndex(INDEX_NAME);

        for (int i = 0; i < deleteDocs; i++) {
            deleteKnnDoc(INDEX_NAME, Integer.toString(i));
        }
        refreshIndex(INDEX_NAME);
        flushIndex(INDEX_NAME);

        waitForReplicatedDocVisibility(INDEX_NAME, expectedVisibleDocs);

        final String query = buildKnnQuery(vector, docsInIndex);
        assertKnnSearchResultCount(INDEX_NAME, query, "_primary", expectedVisibleDocs);

        if (ensureMinDataNodesCountForTestingQueriesOnReplica()) {
            assertKnnSearchResultCount(INDEX_NAME, query, "_replica", expectedVisibleDocs);
        }
    }

    private void waitForReplicatedDocVisibility(final String indexName, final int expectedVisibleDocs) throws Exception {
        ensureGreen(indexName);
        assertBusy(() -> {
            assertEquals(
                "Primary shard doc count mismatch. " + buildReplicationDebugInfo(indexName),
                expectedVisibleDocs,
                getDocCountWithPreference(indexName, "_primary")
            );

            if (canTestOnReplicas()) {
                assertEquals(
                    "Replica shard doc count mismatch. " + buildReplicationDebugInfo(indexName),
                    expectedVisibleDocs,
                    getDocCountWithPreference(indexName, "_replica")
                );
                assertSegmentReplicationCheckpointsCaughtUp(indexName);
            }
        }, ASSERT_BUSY_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    }

    private void assertKnnSearchResultCount(
        final String indexName,
        final String query,
        final String preference,
        final int expectedVisibleDocs
    ) throws Exception {
        assertBusy(() -> {
            doKnnWarmup(List.of(indexName));
            final List<KNNResult> knnResults = executeKnnSearch(indexName, query, preference);
            assertEquals(
                "KNN search result count mismatch on "
                    + preference
                    + ". expectedVisibleDocs="
                    + expectedVisibleDocs
                    + ", actualVisibleDocs="
                    + knnResults.size()
                    + ", resultDocIds="
                    + knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toList())
                    + ", "
                    + buildReplicationDebugInfo(indexName),
                expectedVisibleDocs,
                knnResults.size()
            );
        }, ASSERT_BUSY_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    }

    @SuppressWarnings("unchecked")
    private void assertSegmentReplicationCheckpointsCaughtUp(final String indexName) throws Exception {
        final List<Map<String, Object>> replicationStats = getSegmentReplicationStats(indexName);
        assertFalse(
            "Expected segment replication stats but none were returned. " + buildReplicationDebugInfo(indexName),
            replicationStats.isEmpty()
        );
        for (Map<String, Object> replicationStat : replicationStats) {
            final int checkpointsBehind = Integer.parseInt(String.valueOf(replicationStat.get("checkpoints_behind")));
            assertEquals(
                "Replica replication checkpoints have not caught up. replicationStat="
                    + replicationStat
                    + ", "
                    + buildReplicationDebugInfo(indexName),
                0,
                checkpointsBehind
            );
        }
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> getSegmentReplicationStats(final String indexName) throws Exception {
        final Request request = new Request("GET", "/_cat/segment_replication/" + indexName);
        request.addParameter("format", "json");
        request.addParameter("h", "index,shard,checkpoints_behind,bytes_behind,current_lag");

        final Response response = client().performRequest(request);
        assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );

        final String responseBody = EntityUtils.toString(response.getEntity());
        if (responseBody.isBlank()) {
            return List.of();
        }

        final List<Object> rawStats = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).list();
        final List<Map<String, Object>> replicationStats = new ArrayList<>(rawStats.size());
        for (Object rawStat : rawStats) {
            replicationStats.add((Map<String, Object>) rawStat);
        }
        return replicationStats;
    }

    private List<KNNResult> executeKnnSearch(final String indexName, final String query, final String preference) throws Exception {
        final Response searchResponse = performSearch(indexName, query, "preference=" + preference);
        final String responseBody = EntityUtils.toString(searchResponse.getEntity());
        return parseSearchResponse(responseBody, FIELD_NAME);
    }

    private int getDocCountWithPreference(final String indexName, final String preference) throws Exception {
        final Request request = new Request("GET", "/" + indexName + "/_count");
        request.addParameter("preference", preference);

        final Response response = client().performRequest(request);
        assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );

        final String responseBody = EntityUtils.toString(response.getEntity());
        final Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        return (Integer) responseMap.get("count");
    }

    private String buildKnnQuery(final Float[] vector, final int k) throws IOException {
        final XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilder.startObject("knn");
        queryBuilder.startObject(FIELD_NAME);
        queryBuilder.field("vector", vector);
        queryBuilder.field("k", k);
        queryBuilder.endObject().endObject().endObject().endObject();
        return queryBuilder.toString();
    }

    @SuppressWarnings("unchecked")
    private String buildReplicationDebugInfo(final String indexName) throws Exception {
        final StringBuilder debugInfo = new StringBuilder();
        debugInfo.append("index=").append(indexName);
        debugInfo.append(", clusterHealth=").append(getIndexClusterHealth(indexName));
        debugInfo.append(", primaryDocCount=").append(getDocCountWithPreference(indexName, "_primary"));

        if (canTestOnReplicas()) {
            debugInfo.append(", replicaDocCount=").append(getDocCountWithPreference(indexName, "_replica"));
        }

        debugInfo.append(", deletedDocCount=").append(getDeletedDocCount(indexName));
        debugInfo.append(", shardRouting=").append(getShardRoutingSummary(indexName));
        debugInfo.append(", segmentSummary=").append(getSegmentSummary(indexName));
        if (canTestOnReplicas()) {
            debugInfo.append(", segmentReplicationStats=").append(getSegmentReplicationStats(indexName));
        }
        return debugInfo.toString();
    }

    private String getIndexClusterHealth(final String indexName) throws Exception {
        final Request request = new Request("GET", "/_cluster/health/" + indexName);
        final Response response = client().performRequest(request);
        return EntityUtils.toString(response.getEntity());
    }

    @SuppressWarnings("unchecked")
    private int getDeletedDocCount(final String indexName) throws Exception {
        final Map<String, Object> segmentsResponse = getSegments(indexName);
        final Map<String, Object> indices = (Map<String, Object>) segmentsResponse.get("indices");
        final Map<String, Object> indexData = (Map<String, Object>) indices.get(indexName);
        final Map<String, Object> shards = (Map<String, Object>) indexData.get("shards");

        int deletedDocs = 0;
        for (Object shardList : shards.values()) {
            final List<Map<String, Object>> shardCopies = (List<Map<String, Object>>) shardList;
            for (Map<String, Object> shardCopy : shardCopies) {
                final Map<String, Object> segments = (Map<String, Object>) shardCopy.get("segments");
                for (Object segmentData : segments.values()) {
                    final Map<String, Object> segment = (Map<String, Object>) segmentData;
                    deletedDocs += ((Number) segment.get("deleted_docs")).intValue();
                }
            }
        }
        return deletedDocs;
    }

    @SuppressWarnings("unchecked")
    private String getShardRoutingSummary(final String indexName) throws Exception {
        final Request request = new Request("GET", "/_cluster/state/routing_table/" + indexName);
        final Response response = client().performRequest(request);
        final String responseBody = EntityUtils.toString(response.getEntity());
        final Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        final Map<String, Object> routingTable = (Map<String, Object>) responseMap.get("routing_table");
        final Map<String, Object> indices = (Map<String, Object>) routingTable.get("indices");
        final Map<String, Object> indexRouting = (Map<String, Object>) indices.get(indexName);
        final Map<String, Object> shards = (Map<String, Object>) indexRouting.get("shards");

        final List<String> shardSummaries = new ArrayList<>();
        for (Map.Entry<String, Object> shardEntry : shards.entrySet()) {
            final List<Map<String, Object>> shardCopies = (List<Map<String, Object>>) shardEntry.getValue();
            for (Map<String, Object> shardRouting : shardCopies) {
                shardSummaries.add(
                    "shard="
                        + shardEntry.getKey()
                        + ",primary="
                        + shardRouting.get("primary")
                        + ",state="
                        + shardRouting.get("state")
                        + ",node="
                        + shardRouting.get("node")
                );
            }
        }
        return shardSummaries.toString();
    }

    @SuppressWarnings("unchecked")
    private String getSegmentSummary(final String indexName) throws Exception {
        final Map<String, Object> segmentsResponse = getSegments(indexName);
        final Map<String, Object> indices = (Map<String, Object>) segmentsResponse.get("indices");
        final Map<String, Object> indexData = (Map<String, Object>) indices.get(indexName);
        final Map<String, Object> shards = (Map<String, Object>) indexData.get("shards");

        final List<String> summaries = new ArrayList<>();
        for (Map.Entry<String, Object> shardEntry : shards.entrySet()) {
            final List<Map<String, Object>> shardCopies = (List<Map<String, Object>>) shardEntry.getValue();
            for (Map<String, Object> shardCopy : shardCopies) {
                final Map<String, Object> routing = (Map<String, Object>) shardCopy.get("routing");
                summaries.add(
                    "shard="
                        + shardEntry.getKey()
                        + ",primary="
                        + routing.get("primary")
                        + ",num_search_segments="
                        + shardCopy.get("num_search_segments")
                        + ",max_generation="
                        + getMaxSegmentGeneration(shardCopy)
                );
            }
        }
        return summaries.toString();
    }

    @SuppressWarnings("unchecked")
    private long getMaxSegmentGeneration(final Map<String, Object> shardCopy) {
        final Map<String, Object> segments = (Map<String, Object>) shardCopy.get("segments");
        long maxGeneration = -1;
        for (Object segmentData : segments.values()) {
            final Map<String, Object> segment = (Map<String, Object>) segmentData;
            final long generation = ((Number) segment.get("generation")).longValue();
            maxGeneration = Math.max(maxGeneration, generation);
        }
        return maxGeneration;
    }

    private boolean canTestOnReplicas() {
        return getDataNodeCount() > 1;
    }

    private boolean ensureMinDataNodesCountForTestingQueriesOnReplica() {
        final int dataNodeCount = getDataNodeCount();
        if (dataNodeCount <= 1) {
            log.warn(
                "Not running segment replication tests named: "
                    + "testSearchOnReplicas_whenIndexHasDeletedDocs_thenSuccess, as data nodes count is not atleast 2. "
                    + "Actual datanode count : {}",
                dataNodeCount
            );
            Assert.assertTrue(true);
            return false;
        }
        return true;
    }
}
