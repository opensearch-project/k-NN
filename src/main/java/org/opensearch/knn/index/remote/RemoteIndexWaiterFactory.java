/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

public class RemoteIndexWaiterFactory {

    /**
     * Get the corresponding Waiter implementation for the given RemoteIndexClient. Defaults to Poller for now.
     */
    public static RemoteIndexWaiter getRemoteIndexWaiter(RemoteIndexClient client) {
        return new RemoteIndexPoller(client);
    }
}
