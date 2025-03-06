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
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FAILED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FILE_NAME;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.RUNNING_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.TASK_STATUS;
import static org.opensearch.knn.index.remote.RemoteBuildStatusResponseTests.MOCK_FILE_NAME;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.MOCK_JOB_ID;

public class RemoteIndexPollerTests extends OpenSearchSingleNodeTestCase {
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
        RemoteBuildResponse mockResponse = new RemoteBuildResponse(MOCK_JOB_ID);
        this.mockStatusRequest = new RemoteBuildStatusRequest(mockResponse);
    }

    public void testAwaitVectorBuildTimeout() {
        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMillis(10));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse runningResponse = new RemoteBuildStatusResponse(RUNNING_INDEX_BUILD, null, null);
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

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMillis(100));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse completedResponse = new RemoteBuildStatusResponse(COMPLETED_INDEX_BUILD, MOCK_FILE_NAME, null);
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

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMillis(100));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            String errorMessage = "Failed to build index due to insufficient resources";

            RemoteBuildStatusResponse failedResponse = new RemoteBuildStatusResponse(FAILED_INDEX_BUILD, null, errorMessage);
            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(failedResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            InterruptedException exception = assertThrows(InterruptedException.class, () -> poller.awaitVectorBuild(mockStatusRequest));
            assertTrue(exception.getMessage().contains(errorMessage));

            RemoteBuildStatusResponse failedResponseNoError = new RemoteBuildStatusResponse(FAILED_INDEX_BUILD, null, null);
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

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMillis(100));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse invalidResponse = new RemoteBuildStatusResponse(COMPLETED_INDEX_BUILD, null, null);
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

            when(KNNSettings.getRemoteBuildClientTimeout()).thenReturn(TimeValue.timeValueMillis(100));
            when(KNNSettings.getRemoteBuildClientPollInterval()).thenReturn(TimeValue.timeValueMillis(10));

            RemoteBuildStatusResponse invalidResponse = new RemoteBuildStatusResponse(null, null, null);
            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            when(mockClient.getBuildStatus(mockStatusRequest)).thenReturn(invalidResponse);
            IOException exception = assertThrows(IOException.class, () -> poller.awaitVectorBuild(mockStatusRequest));
            assertEquals("Invalid response format, missing " + TASK_STATUS, exception.getMessage());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
