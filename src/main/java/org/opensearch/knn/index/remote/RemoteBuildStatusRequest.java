/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Getter;

/**
 * Request object to extract and wrap necessary parameters for the `/_status` API.
 */
@Getter
public class RemoteBuildStatusRequest {
    private final String jobId;

    public RemoteBuildStatusRequest(RemoteBuildResponse remoteBuildResponse) {
        this.jobId = remoteBuildResponse.getJobId();
    }
}
