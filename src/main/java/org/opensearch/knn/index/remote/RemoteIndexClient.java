/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteStatusResponse;

import java.io.IOException;

/**
 * Interface which dictates how we interact with a remote index build service.
 */
public interface RemoteIndexClient {

    /**
     * Submit a build to the Remote Vector Build Service.
     * @return RemoteBuildResponse from the server
     */
    RemoteBuildResponse submitVectorBuild(RemoteBuildRequest remoteBuildRequest) throws IOException;

    /**
     * Await the completion of the index build and for the server to return the path to the completed index
     * @param remoteBuildResponse  the /_build request response from the server
     * @return remoteStatusResponse from the server
     */
    RemoteStatusResponse awaitVectorBuild(RemoteBuildResponse remoteBuildResponse);

    /**
     * Construct the RemoteBuildRequest from the given parameters
     * @param indexSettings IndexSettings to use to get the repository metadata
     * @param indexInfo BuildIndexParams to use to get the index info
     * @param repositoryMetadata RepositoryMetadata to use to get the repository type
     * @param blobName blob name to use to get the blob name
     * @return RemoteBuildRequest to use to submit the build
     * @throws IOException if there is an error constructing the request
     */
    RemoteBuildRequest constructBuildRequest(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException;
}
