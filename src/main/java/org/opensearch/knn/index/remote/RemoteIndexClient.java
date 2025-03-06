/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import java.io.IOException;

/**
 * Interface which dictates how we interact with a remote index build service.
 */
public interface RemoteIndexClient {

    /**
     * Submit a build to the Remote Vector Build Service.
     * @return RemoteBuildResponse from the server
     * @throws IOException if there is an error communicating with the server
     */
    RemoteBuildResponse submitVectorBuild(RemoteBuildRequest remoteBuildRequest) throws IOException;

    /**
     * Get the status of an index build
     * @param remoteBuildStatusRequest the status request object containing the job ID to check
     * @return remoteStatusResponse from the server
     * @throws IOException if there is an error communicating with the server
     */
    RemoteBuildStatusResponse getBuildStatus(RemoteBuildStatusRequest remoteBuildStatusRequest) throws IOException;
}
