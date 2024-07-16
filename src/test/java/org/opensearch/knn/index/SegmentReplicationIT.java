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
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;

import java.util.List;

/**
 * This IT class contains will contain special cases of IT for segment replication behavior.
 * All the index created in this test will have replication type SEGMENT, number of replicas: 1 and should be run on
 * at-least 2 node configuration.
 */
@Log4j2
public class SegmentReplicationIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "segment-replicated-knn-index";

    @SneakyThrows
    public void testSearchOnReplicas_whenIndexHasDeletedDocs_thenSuccess() {
        createKnnIndex(INDEX_NAME, getKNNSegmentReplicatedIndexSettings(), createKNNIndexMethodFieldMapping(FIELD_NAME, 2));

        Float[] vector = { 1.3f, 2.2f };
        int docsInIndex = 10;

        for (int i = 0; i < docsInIndex; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
        }
        refreshIndex(INDEX_NAME);
        int deleteDocs = 5;
        for (int i = 0; i < deleteDocs; i++) {
            deleteKnnDoc(INDEX_NAME, Integer.toString(i));
        }
        refreshIndex(INDEX_NAME);
        // sleep for 5sec to ensure data is replicated. I don't have a better way here to know if segments has been
        // replicated.
        Thread.sleep(5000);
        // validate warmup is successful or not.
        doKnnWarmup(List.of(INDEX_NAME));

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilder.startObject("knn");
        queryBuilder.startObject(FIELD_NAME);
        queryBuilder.field("vector", vector);
        queryBuilder.field("k", docsInIndex);
        queryBuilder.endObject().endObject().endObject().endObject();

        // validate primaries are working
        Response searchResponse = performSearch(INDEX_NAME, queryBuilder.toString(), "preference=_primary");
        String responseBody = EntityUtils.toString(searchResponse.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(docsInIndex - deleteDocs, knnResults.size());

        if (ensureMinDataNodesCountForTestingQueriesOnReplica()) {
            // validate replicas are working
            searchResponse = performSearch(INDEX_NAME, queryBuilder.toString(), "preference=_replica");
            responseBody = EntityUtils.toString(searchResponse.getEntity());
            knnResults = parseSearchResponse(responseBody, FIELD_NAME);
            assertEquals(docsInIndex - deleteDocs, knnResults.size());
        }
    }

    private boolean ensureMinDataNodesCountForTestingQueriesOnReplica() {
        int dataNodeCount = getDataNodeCount();
        if (dataNodeCount <= 1) {
            log.warn(
                "Not running segment replication tests named: "
                    + "testSearchOnReplicas_whenIndexHasDeletedDocs_thenSuccess, as data nodes count is not atleast 2. "
                    + "Actual datanode count : {}",
                dataNodeCount
            );
            Assert.assertTrue(true);
            // making the test successful because we don't want to break already running tests.
            return false;
        }
        return true;
    }
}
