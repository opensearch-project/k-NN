/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Getter;

import java.net.URI;

/**
 * HTTP-specific implementation of RemoteBuildResponse to pass the endpoint back to awaitVectorBuild
 */
@Getter
public class HTTPRemoteBuildResponse implements RemoteBuildResponse {
    private final String jobId;
    private final URI endpoint;

    public HTTPRemoteBuildResponse(String requestId, URI endpoint) {
        this.jobId = requestId;
        this.endpoint = endpoint;
    }
}
