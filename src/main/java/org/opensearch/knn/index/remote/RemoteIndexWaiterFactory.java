/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.remoteindexbuild.client.RemoteIndexClient;

import java.util.function.Supplier;

public class RemoteIndexWaiterFactory {

    /**
     * Get the corresponding Waiter implementation for the given RemoteIndexClient. Defaults to Poller for now.
     * @param client the remote index client used to check build status
     * @param isMergeAborted supplier that returns true if the merge has been aborted; enables early termination
     *                       of polling when the merge is no longer needed
     */
    public static RemoteIndexWaiter getRemoteIndexWaiter(RemoteIndexClient client, Supplier<Boolean> isMergeAborted) {
        return new RemoteIndexPoller(client, isMergeAborted);
    }
}
