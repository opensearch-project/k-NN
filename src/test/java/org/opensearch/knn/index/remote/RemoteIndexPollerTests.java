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

import java.util.HashSet;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL_SECONDS_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_TIMEOUT_MINUTES_SETTING;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.COMPLETED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.FAILED_INDEX_BUILD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.MOCK_FILE_NAME;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.RUNNING_INDEX_BUILD;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.MOCK_JOB_ID;

public class RemoteIndexPollerTests extends OpenSearchSingleNodeTestCase {
    @Mock
    protected static ClusterService clusterService;

    @Mock
    private RemoteIndexClient mockClient;

    protected AutoCloseable openMocks;
    private RemoteBuildResponse mockResponse;

    @Before
    public void setup() {
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));

        mockClient = mock(RemoteIndexClient.class);
        mockResponse = new RemoteBuildResponse(MOCK_JOB_ID);
    }

    public void testAwaitVectorBuildTimeout() {
        KNNSettings knnSettingsMock = mock(KNNSettings.class);

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_CLIENT_TIMEOUT_MINUTES_SETTING.getKey())).thenReturn(
                TimeValue.timeValueMillis(100)
            );
            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL_SECONDS_SETTING.getKey())).thenReturn(
                TimeValue.timeValueMillis(10)
            );

            RemoteBuildStatusResponse runningResponse = new RemoteBuildStatusResponse(RUNNING_INDEX_BUILD, null, null);
            when(mockClient.getBuildStatus(new RemoteBuildResponse(MOCK_JOB_ID))).thenReturn(runningResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            assertThrows(InterruptedException.class, () -> poller.pollRemoteEndpoint(mockResponse));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void testAwaitVectorBuildCompleted() {
        KNNSettings knnSettingsMock = mock(KNNSettings.class);

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_CLIENT_TIMEOUT_MINUTES_SETTING.getKey())).thenReturn(
                TimeValue.timeValueSeconds(5)
            );
            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL_SECONDS_SETTING.getKey())).thenReturn(
                TimeValue.timeValueMillis(10)
            );

            RemoteBuildStatusResponse completedResponse = new RemoteBuildStatusResponse(COMPLETED_INDEX_BUILD, MOCK_FILE_NAME, null);
            when(mockClient.getBuildStatus(new RemoteBuildResponse(MOCK_JOB_ID))).thenReturn(completedResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            RemoteBuildStatusResponse response = poller.pollRemoteEndpoint(mockResponse);
            assertEquals(COMPLETED_INDEX_BUILD, response.getTaskStatus());
            assertEquals(MOCK_FILE_NAME, response.getFileName());
            assertNull(response.getErrorMessage());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void testAwaitVectorBuildFailed() {
        KNNSettings knnSettingsMock = mock(KNNSettings.class);

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_CLIENT_TIMEOUT_MINUTES_SETTING.getKey())).thenReturn(
                TimeValue.timeValueSeconds(5)
            );
            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL_SECONDS_SETTING.getKey())).thenReturn(
                TimeValue.timeValueMillis(10)
            );

            String errorMessage = "Failed to build index due to insufficient resources";
            RemoteBuildStatusResponse failedResponse = new RemoteBuildStatusResponse(FAILED_INDEX_BUILD, null, errorMessage);
            when(mockClient.getBuildStatus(new RemoteBuildResponse(MOCK_JOB_ID))).thenReturn(failedResponse);

            RemoteIndexPoller poller = new RemoteIndexPoller(mockClient);

            InterruptedException exception = assertThrows(InterruptedException.class, () -> poller.pollRemoteEndpoint(mockResponse));
            assertTrue(exception.getMessage().contains(errorMessage));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
