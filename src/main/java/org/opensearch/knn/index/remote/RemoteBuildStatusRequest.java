/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Request object to extract and wrap necessary parameters for the `/_status` API.
 */
@Getter
@AllArgsConstructor
public class RemoteBuildStatusRequest {
    private final String jobId;

    /**
     * Creates a status request from a build response using its job ID.
     */
    public static RemoteBuildStatusRequest build(RemoteBuildResponse remoteBuildResponse) {
        return new RemoteBuildStatusRequest(remoteBuildResponse.getJobId());
    }
}
