/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.Builder;
import lombok.Getter;

/**
 * Request object to extract and wrap necessary parameters for the `/_status` API.
 */
@Getter
@Builder
public class RemoteBuildStatusRequest {
    private final String jobId;
}
