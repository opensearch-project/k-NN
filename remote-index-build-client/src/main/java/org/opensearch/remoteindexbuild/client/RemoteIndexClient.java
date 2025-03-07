/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.client;

import org.opensearch.remoteindexbuild.model.RemoteBuildRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildResponse;
import org.opensearch.remoteindexbuild.model.RemoteStatusResponse;

import java.io.IOException;

/**
 * Interface which dictates how we interact with a remote index build service.
 */
public interface RemoteIndexClient {

    String BUILD_ENDPOINT = "/_build";
    String STATUS_ENDPOINT = "/_status";

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
}
