/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

/**
 * Generic remote build response interface
 */
public interface RemoteBuildResponse {
    String getJobId();
}
