/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.*;
import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;

public class RestoreSnapshotIT extends OpenSearchRestTestCase {

    private String index;
    private String snapshot;

    public void setupSnapshotRestore() throws Exception {
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

        // create index
        XContentBuilder settings = jsonBuilder();
        settings.startObject();
        {
            settings.startObject("settings");
            settings.field(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1);
            settings.field(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 1);
            settings.field(IndexSettings.INDEX_SOFT_DELETES_SETTING.getKey(), true);
            settings.endObject();
        }
        settings.endObject();

        Request createIndex = new Request("PUT", "/" + index);
        createIndex.setJsonEntity(settings.toString());
        createIndex.setOptions(allowTypesRemovalWarnings());
        client().performRequest(createIndex);

        // create repo
        XContentBuilder repoConfig = JsonXContent.contentBuilder().startObject();
        {
            repoConfig.field("type", "fs");
            repoConfig.startObject("settings");
            {
                repoConfig.field("compress", randomBoolean());
                repoConfig.field("location", pathRepo);
            }
            repoConfig.endObject();
        }
        repoConfig.endObject();
        Request createRepoRequest = new Request("PUT", "/_snapshot/repo");
        createRepoRequest.setJsonEntity(repoConfig.toString());
        client().performRequest(createRepoRequest);

        // create snapshot
        Request createSnapshot = new Request("PUT", "/_snapshot/repo/" + snapshot);
        createSnapshot.addParameter("wait_for_completion", "true");
        createSnapshot.setJsonEntity("{\"indices\": \"" + index + "\"}");
        client().performRequest(createSnapshot);
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
        Request restoreRequest = new Request("POST", "/_snapshot/repo/" + snapshot + "/_restore");
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
        Request restoreRequest = new Request("POST", "/_snapshot/repo/" + snapshot + "/_restore");
        restoreRequest.addParameter("wait_for_completion", "true");
        restoreRequest.setJsonEntity(restoreCommand.toString());
        final ResponseException error = expectThrows(ResponseException.class, () -> client().performRequest(restoreRequest));
        assertThat(error.getMessage(), containsString("cannot remove UnmodifiableOnRestore setting [index.knn] on restore"));
    }
}
