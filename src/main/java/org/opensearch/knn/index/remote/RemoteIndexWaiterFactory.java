/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

public class RemoteIndexWaiterFactory {
    // Default to poller
    public static RemoteIndexWaiter getRemoteIndexWaiter(RemoteIndexClient client) {
        return new RemoteIndexPoller(client);
    }
}
