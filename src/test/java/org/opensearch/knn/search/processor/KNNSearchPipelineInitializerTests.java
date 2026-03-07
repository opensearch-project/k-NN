/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import org.opensearch.action.search.PutSearchPipelineRequest;
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.ClusterStateListener;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.pipeline.PipelineConfiguration;
import org.opensearch.search.pipeline.SearchPipelineMetadata;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.AdminClient;
import org.opensearch.transport.client.ClusterAdminClient;
import org.mockito.ArgumentCaptor;

import java.util.Collections;
import java.util.Map;

import static org.mockito.Mockito.*;

public class KNNSearchPipelineInitializerTests extends KNNTestCase {

    private Client client;
    private ClusterService clusterService;
    private ClusterAdminClient clusterAdminClient;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        client = mock(Client.class);
        clusterService = mock(ClusterService.class);
        AdminClient adminClient = mock(AdminClient.class);
        clusterAdminClient = mock(ClusterAdminClient.class);
        when(client.admin()).thenReturn(adminClient);
        when(adminClient.cluster()).thenReturn(clusterAdminClient);
    }

    public void testInitialize_nonClusterManagerNode_doesNotAddListener() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "data,ingest").build());

        KNNSearchPipelineInitializer.initialize(client, clusterService);

        verify(clusterService, never()).addListener(any());
    }

    public void testInitialize_clusterManagerNode_addsListener() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager,data,ingest").build());

        KNNSearchPipelineInitializer.initialize(client, clusterService);

        verify(clusterService).addListener(any(ClusterStateListener.class));
    }

    public void testListener_nonClusterManagerEvent_skips() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager").build());
        KNNSearchPipelineInitializer.initialize(client, clusterService);

        ClusterStateListener listener = captureListener();
        ClusterChangedEvent event = mock(ClusterChangedEvent.class);
        when(event.localNodeClusterManager()).thenReturn(false);

        listener.clusterChanged(event);

        verify(clusterAdminClient, never()).putSearchPipeline(any(), any());
        verify(clusterService, never()).removeListener(any());
    }

    public void testListener_pipelineAlreadyExists_removesListener() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager").build());
        KNNSearchPipelineInitializer.initialize(client, clusterService);

        ClusterStateListener listener = captureListener();
        ClusterChangedEvent event = mockEventWithPipeline(true);

        listener.clusterChanged(event);

        verify(clusterService).removeListener(listener);
        verify(clusterAdminClient, never()).putSearchPipeline(any(), any());
    }

    public void testListener_pipelineNotExists_callsPutSearchPipeline() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager").build());
        KNNSearchPipelineInitializer.initialize(client, clusterService);

        ClusterStateListener listener = captureListener();
        ClusterChangedEvent event = mockEventWithPipeline(false);

        listener.clusterChanged(event);

        ArgumentCaptor<PutSearchPipelineRequest> reqCaptor = ArgumentCaptor.forClass(PutSearchPipelineRequest.class);
        verify(clusterAdminClient).putSearchPipeline(reqCaptor.capture(), any());
        assertEquals(KNNSearchPipelineInitializer.KNN_DEFAULT_SEARCH_PIPELINE_NAME, reqCaptor.getValue().getId());
    }

    @SuppressWarnings("unchecked")
    public void testListener_putSucceeds_removesListener() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager").build());
        KNNSearchPipelineInitializer.initialize(client, clusterService);

        ClusterStateListener listener = captureListener();
        ClusterChangedEvent event = mockEventWithPipeline(false);

        // Capture the ActionListener passed to putSearchPipeline
        ArgumentCaptor<ActionListener<AcknowledgedResponse>> actionListenerCaptor = ArgumentCaptor.forClass(ActionListener.class);
        listener.clusterChanged(event);
        verify(clusterAdminClient).putSearchPipeline(any(), actionListenerCaptor.capture());

        // Simulate success
        actionListenerCaptor.getValue().onResponse(new AcknowledgedResponse(true));

        verify(clusterService).removeListener(listener);
    }

    @SuppressWarnings("unchecked")
    public void testListener_putFails_doesNotRemoveListener() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager").build());
        KNNSearchPipelineInitializer.initialize(client, clusterService);

        ClusterStateListener listener = captureListener();
        ClusterChangedEvent event = mockEventWithPipeline(false);

        ArgumentCaptor<ActionListener<AcknowledgedResponse>> actionListenerCaptor = ArgumentCaptor.forClass(ActionListener.class);
        listener.clusterChanged(event);
        verify(clusterAdminClient).putSearchPipeline(any(), actionListenerCaptor.capture());

        // Simulate failure
        actionListenerCaptor.getValue().onFailure(new RuntimeException("test failure"));

        // Listener should NOT be removed so it retries on next cluster state change
        verify(clusterService, never()).removeListener(any());
    }

    @SuppressWarnings("unchecked")
    public void testListener_nullMetadata_callsPutSearchPipeline() {
        when(clusterService.getSettings()).thenReturn(Settings.builder().put("node.roles", "cluster_manager").build());
        KNNSearchPipelineInitializer.initialize(client, clusterService);

        ClusterStateListener listener = captureListener();

        // Event where metadata.custom returns null (no search pipeline metadata at all)
        ClusterChangedEvent event = mock(ClusterChangedEvent.class);
        when(event.localNodeClusterManager()).thenReturn(true);
        ClusterState state = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        when(event.state()).thenReturn(state);
        when(state.metadata()).thenReturn(metadata);
        when(metadata.custom(SearchPipelineMetadata.TYPE)).thenReturn(null);

        listener.clusterChanged(event);

        verify(clusterAdminClient).putSearchPipeline(any(PutSearchPipelineRequest.class), any());
    }

    private ClusterStateListener captureListener() {
        ArgumentCaptor<ClusterStateListener> captor = ArgumentCaptor.forClass(ClusterStateListener.class);
        verify(clusterService).addListener(captor.capture());
        return captor.getValue();
    }

    private ClusterChangedEvent mockEventWithPipeline(boolean pipelineExists) {
        ClusterChangedEvent event = mock(ClusterChangedEvent.class);
        when(event.localNodeClusterManager()).thenReturn(true);
        ClusterState state = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        when(event.state()).thenReturn(state);
        when(state.metadata()).thenReturn(metadata);

        Map<String, PipelineConfiguration> pipelines = pipelineExists
            ? Map.of(KNNSearchPipelineInitializer.KNN_DEFAULT_SEARCH_PIPELINE_NAME, mock(PipelineConfiguration.class))
            : Collections.emptyMap();
        SearchPipelineMetadata pipelineMetadata = mock(SearchPipelineMetadata.class);
        when(pipelineMetadata.getPipelines()).thenReturn(pipelines);
        when(metadata.custom(SearchPipelineMetadata.TYPE)).thenReturn(pipelineMetadata);

        return event;
    }
}
