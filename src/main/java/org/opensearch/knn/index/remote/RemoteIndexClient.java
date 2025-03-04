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
     */
    RemoteBuildResponse submitVectorBuild(RemoteBuildRequest remoteBuildRequest) throws IOException;

    /**
     * Await the completion of the index build and for the server to return the path to the completed index
     *
     * @param remoteBuildResponse the /_build request response from the server
     * @return remoteStatusResponse from the server
     * @throws InterruptedException if the thread is interrupted while waiting for the build to complete
     * @throws IOException if there is an error communicating with the server
     */
    RemoteBuildStatusResponse awaitVectorBuild(RemoteBuildResponse remoteBuildResponse) throws InterruptedException, IOException;
}
