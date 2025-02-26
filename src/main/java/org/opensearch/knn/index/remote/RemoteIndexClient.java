/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

public interface RemoteIndexClient {
    String submitVectorBuild(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException;

    String awaitVectorBuild(String jobId);
}
