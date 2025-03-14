/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.client;

public class RemoteIndexClientFactory {

    // Default to HTTP client
    public static RemoteIndexClient getRemoteIndexClient(final String endpoint) {
        return new RemoteIndexHTTPClient(endpoint);
    }
}
