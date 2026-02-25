/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.remoteindexbuild.client.RemoteIndexClient;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusResponse;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import static org.opensearch.knn.index.remote.RemoteIndexPoller.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.RemoteIndexPoller.FAILED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.RemoteIndexPoller.FILE_NAME;
import static org.opensearch.knn.index.remote.RemoteIndexPoller.RUNNING_INDEX_BUILD;
import static org.opensearch.knn.index.remote.RemoteIndexPoller.TASK_STATUS;

public class RemoteIndexPollerTests extends OpenSearchSingleNodeTestCase {
    public static final String MOCK_JOB_ID = "job-1739930402";
    public static final String MOCK_FILE_NAME = "graph.faiss";

    @Mock
    private static ClusterService clusterService;

    @Mock
    private RemoteIndexClient mockClient;

    AutoCloseable openMocks;
    RemoteBuildStatusRequest mockStatusRequest;
    KNNSettings knnSettingsMock;

    @Before
    public void setup() {
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
        this.knnSettingsMock = mock(KNNSettings.class);

        mockClient = mock(RemoteIndexClient.class);
        this.mockStatusRequest = RemoteBuildStatusRequest.builder().jobId(MOCK_JOB_ID).build();
    }

    public void testAwaitVectorBuildTimeout() {
        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMillis(10));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse runningResponse = RemoteBuildStatusResponse.builder()
                .taskStatus(RUNNING_INDEX_BUILD)
                .fileName(null)
                .errorMessage(null)
                .build();
            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(runningResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            assertThrows(InterruptedException.class, () -> poller.awaitVectorBuild(mockStatusRequest));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void testAwaitVectorBuildCompleted() {
        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMinutes(1));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse completedResponse = RemoteBuildStatusResponse.builder()
                .taskStatus(COMPLETED_INDEX_BUILD)
                .fileName(MOCK_FILE_NAME)
                .errorMessage(null)
                .build();
            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(completedResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            RemoteBuildStatusResponse response = poller.awaitVectorBuild(mockStatusRequest);
            assertEquals(COMPLETED_INDEX_BUILD, response.getTaskStatus());
            assertEquals(MOCK_FILE_NAME, response.getFileName());
            assertNull(response.getErrorMessage());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void testAwaitVectorBuildFailed() {
        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMinutes(1));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            String errorMessage = "Failed to build index due to insufficient resources";

            RemoteBuildStatusResponse failedResponse = RemoteBuildStatusResponse.builder()
                .taskStatus(FAILED_INDEX_BUILD)
                .fileName(null)
                .errorMessage(errorMessage)
                .build();
            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(failedResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            InterruptedException exception = assertThrows(InterruptedException.class, () -> poller.awaitVectorBuild(mockStatusRequest));
            assertTrue(exception.getMessage().contains(errorMessage));

            RemoteBuildStatusResponse failedResponseNoError = RemoteBuildStatusResponse.builder()
                .taskStatus(FAILED_INDEX_BUILD)
                .fileName(null)
                .errorMessage(null)
                .build();

            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(failedResponseNoError);
            InterruptedException exceptionWithoutError = assertThrows(
                InterruptedException.class,
                () -> poller.awaitVectorBuild(mockStatusRequest)
            );
            assertFalse(exceptionWithoutError.getMessage().contains(errorMessage));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void testMissingIndexPathForCompletedStatus() {
        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMinutes(1));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse invalidResponse = RemoteBuildStatusResponse.builder()
                .taskStatus(COMPLETED_INDEX_BUILD)
                .fileName(null)
                .errorMessage(null)
                .build();
            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(invalidResponse);
            IOException exception = assertThrows(IOException.class, () -> poller.awaitVectorBuild(mockStatusRequest));
            assertEquals(
                "Invalid response format, missing " + FILE_NAME + " for " + COMPLETED_INDEX_BUILD + " status",
                exception.getMessage()
            );
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void testMissingTaskStatus() {
        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMinutes(1));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse invalidResponse = RemoteBuildStatusResponse.builder()
                .taskStatus(null)
                .fileName(null)
                .errorMessage(null)
                .build();
            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(invalidResponse);
            IOException exception = assertThrows(IOException.class, () -> poller.awaitVectorBuild(mockStatusRequest));
            assertEquals("Invalid response format, missing " + TASK_STATUS, exception.getMessage());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
