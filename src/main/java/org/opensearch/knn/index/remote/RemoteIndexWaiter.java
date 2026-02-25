/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.remoteindexbuild.model.RemoteBuildStatusRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusResponse;

import java.io.IOException;

public interface RemoteIndexWaiter {

    /**
     * Wait for the remote index to be built and return its response when completed.
     * Implementations can use KNN_REMOTE_BUILD_CLIENT_TIMEOUT to determine when to abandon the build.
     * @param remoteBuildStatusRequest the status request object
     * @return remoteStatusResponse from the server
     * @throws InterruptedException if the waiting process gets interrupted or build fails
     * @throws IOException if there is an error communicating with the server
     */
    RemoteBuildStatusResponse awaitVectorBuild(RemoteBuildStatusRequest remoteBuildStatusRequest) throws InterruptedException, IOException;
}
