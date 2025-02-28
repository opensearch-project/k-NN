/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

public class RemoteIndexClientFactory {

    public static final String TYPE_HTTP = "HTTP";

    public static RemoteIndexClient getRemoteIndexClient(String type) {
        if (TYPE_HTTP.equalsIgnoreCase(type)) {
            return new RemoteIndexHTTPClient();
        }
        throw new IllegalArgumentException("Unsupported RemoteIndexClient type: " + type);
    }
}
