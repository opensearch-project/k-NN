/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

/**
 * Interface which dictates how we interact with a remote index build service.
 */
interface RemoteIndexClient {
    /**
     * Submit an index build request to the build service endpoint.
     * @param indexSettings IndexSettings for the index being built
     * @param indexInfo BuildIndexParams for the index being built
     * @param repositoryMetadata RepositoryMetadata representing the registered repo
     * @param blobName The name of the blob written to the repo, to be suffixed with ".knnvec" or ".knndid"
     * @return job_id from the server response used to track the job
     * @throws IOException if there is an issue with the request
     */
    String submitVectorBuild(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException;

    /**
     * Await the completion of the index build and for the server to return the path to the completed index
     * @param jobId identifier from the server to track the job
     * @return the path to the completed index
     */
    String awaitVectorBuild(String jobId);
}
