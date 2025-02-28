/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

/**
 * Remote build response. Currently, this just contains the jobId from the server.
 * In the future, this may be an interface if different clients expect different responses.
 */
public record RemoteBuildResponse(String jobId) {
}
