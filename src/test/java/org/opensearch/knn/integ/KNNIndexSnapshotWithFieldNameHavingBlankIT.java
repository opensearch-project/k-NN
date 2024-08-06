/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.junit.After;
import org.junit.Assert;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.List;

@Log4j2
@AllArgsConstructor
public class KNNIndexSnapshotWithFieldNameHavingBlankIT extends KNNRestTestCase {
    private static final String FIELD_NAME_HAVING_BLANK = "my vector";
    private static final String REPO_NAME = "fieldWithBlankTestRepo-" + System.nanoTime();
    private static final List<String> SNAPSHOTS = List.of("first", "second");
    private static final String INDEX_NAME = "field-with-blank-test-index-" + System.nanoTime();
    private static final int DIMENSION = 16;

    @After
    public void cleanUp() {
        try {
            deleteSnapshots();
        } catch (Exception e) {
            log.error(e);
        }

        try {
            deleteRepo();
        } catch (Exception e) {
            log.error(e);
        }

        try {
            deleteKNNIndex(INDEX_NAME);
        } catch (Exception e) {
            log.error(e);
        }
    }

    @SneakyThrows
    public void testSnapshotTwice() {
        // Create a KNN index
        final String indexMapping = createKnnIndexMapping(FIELD_NAME_HAVING_BLANK, DIMENSION);
        createKnnIndex(INDEX_NAME, indexMapping);

        // Add vectors
        bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME_HAVING_BLANK, 100, DIMENSION);

        // Create repo
        createRepo();

        // Take the first snapshot
        takeSnapshot(SNAPSHOTS.get(0));

        // Take the second snapshot
        takeSnapshot(SNAPSHOTS.get(1));
    }

    private static void createRepo() throws IOException {
        final Request request = new Request("PUT", "_snapshot/" + REPO_NAME);
        final String jsonBody = "{\n"
            + "  \"type\": \"fs\",\n"
            + "  \"settings\": {\n"
            + "    \"compress\": true,\n"
            + "    \"location\": \"testrepo-"
            + System.nanoTime()
            + "\"\n"
            + "  }\n"
            + "}\n";
        request.setJsonEntity(jsonBody);
        final Response response = client().performRequest(request);
        Assert.assertEquals(response.getStatusLine().getStatusCode(), 200);
    }

    private static void takeSnapshot(final String snapshotName) throws IOException {
        final Request request = new Request("PUT", "_snapshot/" + REPO_NAME + "/" + snapshotName);
        final Response response = client().performRequest(request);
        Assert.assertEquals(response.getStatusLine().getStatusCode(), 200);
    }

    private void deleteSnapshots() {
        for (String snapshot : SNAPSHOTS) {
            try {
                final Request request = new Request("DELETE", "_snapshot/" + REPO_NAME + "/" + snapshot);
                final Response response = client().performRequest(request);
                assertEquals(
                    request.getEndpoint() + ": failed",
                    RestStatus.OK,
                    RestStatus.fromCode(response.getStatusLine().getStatusCode())
                );
            } catch (Exception e) {
                log.error(e);
            }
        }
    }

    private void deleteRepo() throws IOException {
        final Request request = new Request("DELETE", "_snapshot/" + REPO_NAME);
        final Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }
}
