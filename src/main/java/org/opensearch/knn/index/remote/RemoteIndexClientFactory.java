/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

public class RemoteIndexClientFactory {

    // Default to HTTP client
    public static RemoteIndexClient getRemoteIndexClient() {
        return new RemoteIndexHTTPClient();
    }
}
