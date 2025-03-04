/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;

import static org.opensearch.knn.index.remote.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FAILED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.RUNNING_INDEX_BUILD;

class RemoteIndexPoller {
    private final RemoteIndexClient client;

    RemoteIndexPoller(RemoteIndexClient client) {
        this.client = client;
    }

    /**
     * Polls the remote endpoint for the status of the build job until timeout.
     *
     * @param remoteBuildResponse The response from the initial build request
     * @return RemoteBuildStatusResponse containing the path of the completed build job
     * @throws InterruptedException if the thread is interrupted while polling
     * @throws IOException if an I/O error occurs
     */
    @SuppressWarnings("BusyWait")
    RemoteBuildStatusResponse pollRemoteEndpoint(RemoteBuildResponse remoteBuildResponse) throws InterruptedException, IOException {
        long startTime = System.currentTimeMillis();
        long timeout = ((TimeValue) KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_CLIENT_TIMEOUT)).getMillis();
        long pollInterval = ((TimeValue) (KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL)))
            .getMillis();

        // Initial delay to allow build service to process the job and store the ID before getting its status.
        // TODO tune default based on benchmarking
        Thread.sleep(pollInterval * 3);

        while (System.currentTimeMillis() - startTime < timeout) {
            RemoteBuildStatusResponse remoteBuildStatusResponse = client.getBuildStatus(remoteBuildResponse);
            String taskStatus = remoteBuildStatusResponse.getTaskStatus();
            switch (taskStatus) {
                case COMPLETED_INDEX_BUILD:
                    return remoteBuildStatusResponse;
                case FAILED_INDEX_BUILD:
                    String errorMessage = remoteBuildStatusResponse.getErrorMessage();
                    if (errorMessage != null) {
                        throw new InterruptedException("Index build failed: " + errorMessage);
                    }
                    throw new InterruptedException("Index build failed without an error message.");
                case RUNNING_INDEX_BUILD:
                    Thread.sleep(pollInterval);
            }
        }
        throw new InterruptedException("Build timed out, falling back to CPU build.");
    }
}
