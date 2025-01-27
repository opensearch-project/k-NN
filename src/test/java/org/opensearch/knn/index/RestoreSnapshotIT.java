/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.junit.Before;
import org.junit.Test;
import lombok.SneakyThrows;
import static org.hamcrest.Matchers.containsString;

public class RestoreSnapshotIT extends KNNRestTestCase {

    private String index = "test-index";;
    private String snapshot = "snapshot-" + index;
    private String repository = "repo";

    @Before
    @SneakyThrows
    public void setUp() {
        super.setUp();
        setupSnapshotRestore(index, snapshot, repository);
    }

    @Test
    @SneakyThrows
    public void testKnnSettingIsModifiable_whenRestore_thenSuccess() {
        // valid restore
        XContentBuilder restoreCommand = JsonXContent.contentBuilder().startObject();
        restoreCommand.field("indices", index);
        restoreCommand.field("rename_pattern", index);
        restoreCommand.field("rename_replacement", "restored-" + index);
        restoreCommand.startObject("index_settings");
        {
            restoreCommand.field("knn.model.index.number_of_shards", 1);
        }
        restoreCommand.endObject();
        restoreCommand.endObject();
        Request restoreRequest = new Request("POST", "/_snapshot/" + repository + "/" + snapshot + "/_restore");
        restoreRequest.addParameter("wait_for_completion", "true");
        restoreRequest.setJsonEntity(restoreCommand.toString());

        final Response restoreResponse = client().performRequest(restoreRequest);
        assertEquals(200, restoreResponse.getStatusLine().getStatusCode());
    }

    @Test
    @SneakyThrows
    public void testKnnSettingIsUnmodifiable_whenRestore_thenFailure() {
        // invalid restore
        XContentBuilder restoreCommand = JsonXContent.contentBuilder().startObject();
        restoreCommand.field("indices", index);
        restoreCommand.field("rename_pattern", index);
        restoreCommand.field("rename_replacement", "restored-" + index);
        restoreCommand.startObject("index_settings");
        {
            restoreCommand.field("index.knn", false);
        }
        restoreCommand.endObject();
        restoreCommand.endObject();
        Request restoreRequest = new Request("POST", "/_snapshot/" + repository + "/" + snapshot + "/_restore");
        restoreRequest.addParameter("wait_for_completion", "true");
        restoreRequest.setJsonEntity(restoreCommand.toString());
        final ResponseException error = expectThrows(ResponseException.class, () -> client().performRequest(restoreRequest));
        assertThat(error.getMessage(), containsString("cannot modify UnmodifiableOnRestore setting [index.knn]" + " on restore"));
    }

    @Test
    @SneakyThrows
    public void testKnnSettingCanBeIgnored_whenRestore_thenSuccess() {
        // valid restore
        XContentBuilder restoreCommand = JsonXContent.contentBuilder().startObject();
        restoreCommand.field("indices", index);
        restoreCommand.field("rename_pattern", index);
        restoreCommand.field("rename_replacement", "restored-" + index);
        restoreCommand.field("ignore_index_settings", "knn.model.index.number_of_shards");
        restoreCommand.endObject();
        Request restoreRequest = new Request("POST", "/_snapshot/" + repository + "/" + snapshot + "/_restore");
        restoreRequest.addParameter("wait_for_completion", "true");
        restoreRequest.setJsonEntity(restoreCommand.toString());
        final Response restoreResponse = client().performRequest(restoreRequest);
        assertEquals(200, restoreResponse.getStatusLine().getStatusCode());
    }

    @Test
    @SneakyThrows
    public void testKnnSettingCannotBeIgnored_whenRestore_thenFailure() {
        // invalid restore
        XContentBuilder restoreCommand = JsonXContent.contentBuilder().startObject();
        restoreCommand.field("indices", index);
        restoreCommand.field("rename_pattern", index);
        restoreCommand.field("rename_replacement", "restored-" + index);
        restoreCommand.field("ignore_index_settings", "index.knn");
        restoreCommand.endObject();
        Request restoreRequest = new Request("POST", "/_snapshot/" + repository + "/" + snapshot + "/_restore");
        restoreRequest.addParameter("wait_for_completion", "true");
        restoreRequest.setJsonEntity(restoreCommand.toString());
        final ResponseException error = expectThrows(ResponseException.class, () -> client().performRequest(restoreRequest));
        assertThat(error.getMessage(), containsString("cannot remove UnmodifiableOnRestore setting [index.knn] on restore"));
    }
}
