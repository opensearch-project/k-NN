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
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.KNNCompressionRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;

/**
 * This IT class contains will contain special cases of IT for segment replication behavior.
 * All the index created in this test will have replication type SEGMENT, number of replicas: 1 and should be run on
 * at-least 2 node configuration.
 */
@Log4j2
public class SegmentReplicationIT extends KNNCompressionRestTestCase {
    private static final String INDEX_NAME = "segment-replicated-knn-index";

    public SegmentReplicationIT(CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testSearchOnReplicas_whenIndexHasDeletedDocs_thenSuccess() {
        createKnnIndex(INDEX_NAME, getKNNSegmentReplicatedIndexSettings(), createFieldMapping(2));

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
            // Segment replication is asynchronous, so wait on the replication signal itself rather than on the k-NN
            // assertion. Polling the assertion under test would retry away a genuinely invalid remote-built graph,
            // which is the exact failure this test exists to catch.
            assertBusy(() -> assertEquals(docsInIndex - deleteDocs, getDocCountWithPreference(INDEX_NAME, "_replica")));

            // validate replicas are working. Asserted once: past this point the replica has the primary's live docs,
            // so a wrong result count is a real failure and must surface immediately.
            searchResponse = performSearch(INDEX_NAME, queryBuilder.toString(), "preference=_replica");
            responseBody = EntityUtils.toString(searchResponse.getEntity());
            knnResults = parseSearchResponse(responseBody, FIELD_NAME);
            assertEquals(docsInIndex - deleteDocs, knnResults.size());
        }
    }

    /**
     * Returns the number of live documents visible to the shard copy selected by {@code preference}. Used to wait for
     * segment replication to deliver the primary's latest commit to the replica before searching it.
     */
    private int getDocCountWithPreference(final String indexName, final String preference) throws Exception {
        final Request request = new Request("GET", "/" + indexName + "/_count");
        request.addParameter("preference", preference);

        final Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        final String responseBody = EntityUtils.toString(response.getEntity());
        final Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        return (Integer) responseMap.get("count");
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

    private String createFieldMapping(int dimension) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension);
        addCompressionMappingFields(builder);
        builder.startObject(KNN_METHOD).field(NAME, METHOD_HNSW).endObject().endObject().endObject().endObject();
        return builder.toString();
    }
}
