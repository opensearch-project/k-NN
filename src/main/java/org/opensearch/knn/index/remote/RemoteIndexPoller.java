/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Setter;
import org.apache.commons.lang.StringUtils;
import org.opensearch.remoteindexbuild.client.RemoteIndexClient;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusResponse;

import java.io.IOException;
import java.time.Duration;
import java.util.Random;

import static org.opensearch.knn.index.KNNSettings.getRemoteBuildClientPollInterval;
import static org.opensearch.knn.index.KNNSettings.getRemoteBuildClientTimeout;

/**
 * Implementation of a {@link RemoteIndexWaiter} that awaits the vector build by polling.
 */
public class RemoteIndexPoller implements RemoteIndexWaiter {
    // The poller waits KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL * INITIAL_DELAY_FACTOR before sending the first status request
    private static final int INITIAL_DELAY_FACTOR = 3;
    private static final double JITTER_LOWER = 0.8;
    private static final double JITTER_UPPER = 1.2;
    public static final String TASK_STATUS = "task_status";
    public static final String FILE_NAME = "file_name";
    public static final String RUNNING_INDEX_BUILD = "RUNNING_INDEX_BUILD";
    public static final String COMPLETED_INDEX_BUILD = "COMPLETED_INDEX_BUILD";
    public static final String FAILED_INDEX_BUILD = "FAILED_INDEX_BUILD";

    private final RemoteIndexClient client;
    private final Random random;
    @Setter
    private long timeout; // Timeout in nanoseconds, to use same units as System.nanoTime().
    @Setter
    private long pollInterval; // Poll interval in milliseconds

    RemoteIndexPoller(RemoteIndexClient client) {
        this.client = client;
        this.random = new Random();
        this.timeout = getRemoteBuildClientTimeout().getNanos();
        this.pollInterval = getRemoteBuildClientPollInterval().getMillis();
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
                case COMPLETED_INDEX_BUILD: {
                    if (StringUtils.isBlank(remoteBuildStatusResponse.getFileName())) {
                        throw new IOException(String.format("Invalid response format, missing %s for %s status", FILE_NAME, taskStatus));
                    }
                    return remoteBuildStatusResponse;
                }
                case FAILED_INDEX_BUILD: {
                    String errorMessage = remoteBuildStatusResponse.getErrorMessage();
                    Duration d = Duration.ofNanos(System.nanoTime() - startTime);
                    throw new InterruptedException(
                            String.format("Remote index build failed after %d minutes. %s", d.toMinutesPart(), errorMessage)
                    );
                }
                case RUNNING_INDEX_BUILD:
                    sleepWithJitter(pollInterval);
                    break;
                default:
                    throw new IOException(String.format("Server returned invalid task status %s", taskStatus));
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

    /**
     * Sleeps for the given {@code baseInterval} with jitter. Example with {@code baseInterval} = 5000ms (5 seconds)
     * <p>
     * If random.nextDouble() = 0.2, sleep for 4.4 seconds
     * <p>
     * If random.nextDouble() = 0.5, sleep for 5 seconds
     * <p>
     * If random.nextDouble() = 0.8, sleep for 5.6 seconds
     * <p>
     * The result will always be between {@code baseInterval * JITTER_LOWER} (inclusive) and {@code baseInterval * JITTER_UPPER} (exclusive)
     *
     * @param baseInterval The base interval in milliseconds
     * @throws InterruptedException if the thread is interrupted while sleeping
     */
    private void sleepWithJitter(long baseInterval) throws InterruptedException {
        long intervalWithJitter = (long) (baseInterval * (JITTER_LOWER + (random.nextDouble() * (JITTER_UPPER - JITTER_LOWER))));
        Thread.sleep(intervalWithJitter);
    }
}
