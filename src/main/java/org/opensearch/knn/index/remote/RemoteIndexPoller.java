/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.time.Duration;
import java.util.Random;

import static org.opensearch.knn.index.remote.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FAILED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FILE_NAME;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.RUNNING_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.TASK_STATUS;

/**
 * Implementation of a {@link RemoteIndexWaiter} that awaits the vector build by polling.
 */
class RemoteIndexPoller implements RemoteIndexWaiter {
    // The poller waits KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL * INITIAL_DELAY_FACTOR before sending the first status request
    private static final int INITIAL_DELAY_FACTOR = 3;

    private static final double JITTER_LOWER = 0.8;
    private static final double JITTER_UPPER = 1.2;

    private final RemoteIndexClient client;
    private final Random random = new Random();

    RemoteIndexPoller(RemoteIndexClient client) {
        this.client = client;
    }

    /**
     * Polls the remote endpoint for the status of the build job until timeout.
     *
     * @param remoteBuildStatusRequest The response from the initial build request
     * @return RemoteBuildStatusResponse containing the path of the completed build job
     * @throws InterruptedException if the thread is interrupted while polling
     * @throws IOException if an I/O error occurs
     */
    public RemoteBuildStatusResponse awaitVectorBuild(RemoteBuildStatusRequest remoteBuildStatusRequest) throws InterruptedException,
        IOException {
        long startTime = System.nanoTime();
        long timeout = KNNSettings.getRemoteBuildClientTimeout().getNanos();
        // Thread.sleep expects millis
        long pollInterval = KNNSettings.getRemoteBuildClientPollInterval().getMillis();

        // Initial delay to allow build service to process the job and store the ID before getting its status.
        // TODO tune default based on benchmarking
        sleepWithJitter(pollInterval * INITIAL_DELAY_FACTOR);

        while (System.nanoTime() - startTime < timeout) {
            RemoteBuildStatusResponse remoteBuildStatusResponse = client.getBuildStatus(remoteBuildStatusRequest);
            String taskStatus = remoteBuildStatusResponse.getTaskStatus();
            if (StringUtils.isBlank(taskStatus)) {
                throw new IOException(String.format("Invalid response format, missing %s", TASK_STATUS));
            }
            switch (taskStatus) {
                case COMPLETED_INDEX_BUILD -> {
                    if (StringUtils.isBlank(remoteBuildStatusResponse.getFileName())) {
                        throw new IOException(String.format("Invalid response format, missing %s for %s status", FILE_NAME, taskStatus));
                    }
                    return remoteBuildStatusResponse;
                }
                case FAILED_INDEX_BUILD -> {
                    String errorMessage = remoteBuildStatusResponse.getErrorMessage();
                    Duration d = Duration.ofNanos(System.nanoTime() - startTime);
                    throw new InterruptedException(
                        String.format("Remote index build failed after %d minutes. %s", d.toMinutesPart(), errorMessage)
                    );
                }
                case RUNNING_INDEX_BUILD -> sleepWithJitter(pollInterval);
                default -> throw new IOException(String.format("Server returned invalid task status %s", taskStatus));
            }
        }
        Duration waitedDuration = Duration.ofNanos(System.nanoTime() - startTime);
        Duration timeoutDuration = Duration.ofNanos(timeout);
        throw new InterruptedException(
            String.format(
                "Remote index build timed out after %d minutes, timeout is set to %d minutes. Falling back to CPU build",
                waitedDuration.toMinutesPart(),
                timeoutDuration.toMinutesPart()
            )
        );
    }

    private void sleepWithJitter(long baseInterval) throws InterruptedException {
        long intervalWithJitter = (long) (baseInterval * (JITTER_LOWER + (random.nextDouble() * (JITTER_UPPER - JITTER_LOWER))));
        Thread.sleep(intervalWithJitter);
    }
}
