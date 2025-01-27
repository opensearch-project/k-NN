/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.hasSize;

public class RestoreSnapshotIT extends OpenSearchRestTestCase {

    private String index;
    private String snapshot;
    private String repository;

    private void setupSnapshotRestore() throws Exception {
        Request clusterSettingsRequest = new Request("GET", "/_cluster/settings");
        clusterSettingsRequest.addParameter("include_defaults", "true");
        Response clusterSettingsResponse = client().performRequest(clusterSettingsRequest);
        Map<String, Object> clusterSettings = entityAsMap(clusterSettingsResponse);

        @SuppressWarnings("unchecked")
        List<String> pathRepos = (List<String>) XContentMapValues.extractValue("defaults.path.repo", clusterSettings);
        assertThat(pathRepos, notNullValue());
        assertThat(pathRepos, hasSize(1));

        final String pathRepo = pathRepos.get(0);

        index = "test-index";
        snapshot = "snapshot-" + index;
        repository = "repo";

        // create index
        Settings indexSettings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 1).put("index.knn", true).build();
        createIndex(index, indexSettings);

        // create repo
        Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
        registerRepository(repository, "fs", true, repoSettings);

        // create snapshot
        createSnapshot(repository, snapshot, true);
    }

    public void testUnmodifiableOnRestoreSettingModifiedOnRestore() throws Exception {
        setupSnapshotRestore();

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

    public void testUnmodifiableOnRestoreSettingIgnoredOnRestore() throws Exception {
        setupSnapshotRestore();

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
