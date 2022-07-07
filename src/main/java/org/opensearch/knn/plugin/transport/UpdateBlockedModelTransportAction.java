/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.action.support.master.TransportMasterNodeAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.ClusterStateTaskConfig;
import org.opensearch.cluster.ClusterStateTaskExecutor;
import org.opensearch.cluster.ClusterStateTaskListener;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.Priority;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.knn.indices.ModelGraveyard;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import lombok.AllArgsConstructor;

import static org.opensearch.knn.common.KNNConstants.PLUGIN_NAME;

/**
 * Transport action used to update blocked modelIds (ModelGraveyard) on the cluster manager node.
 */
@Log4j2
public class UpdateBlockedModelTransportAction extends TransportMasterNodeAction<UpdateBlockedModelRequest, AcknowledgedResponse> {
    private UpdateBlockedModelExecutor updateBlockedModelExecutor;

    @Inject
    public UpdateBlockedModelTransportAction(
        TransportService transportService,
        ClusterService clusterService,
        ThreadPool threadPool,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(
            UpdateBlockedModelAction.NAME,
            transportService,
            clusterService,
            threadPool,
            actionFilters,
            UpdateBlockedModelRequest::new,
            indexNameExpressionResolver
        );
        this.updateBlockedModelExecutor = new UpdateBlockedModelExecutor();
    }

    @Override
    protected String executor() {
        return ThreadPool.Names.SAME;
    }

    @Override
    protected AcknowledgedResponse read(StreamInput streamInput) throws IOException {
        return new AcknowledgedResponse(streamInput);
    }

    @Override
    protected void masterOperation(
        UpdateBlockedModelRequest request,
        ClusterState clusterState,
        ActionListener<AcknowledgedResponse> actionListener
    ) {
        // ClusterManager updates blocked modelIds list based on request parameters
        clusterService.submitStateUpdateTask(
            PLUGIN_NAME,
            new UpdateBlockedModelTask(request.getModelId(), request.isRemoveRequest()),
            ClusterStateTaskConfig.build(Priority.NORMAL),
            updateBlockedModelExecutor,
            new ClusterStateTaskListener() {
                @Override
                public void onFailure(String s, Exception e) {
                    actionListener.onFailure(e);
                }

                @Override
                public void clusterStateProcessed(String source, ClusterState oldState, ClusterState newState) {
                    actionListener.onResponse(new AcknowledgedResponse(true));
                }
            }
        );
    }

    @Override
    protected ClusterBlockException checkBlock(UpdateBlockedModelRequest request, ClusterState clusterState) {
        return null;
    }

    /**
     * UpdateBlockedModelTask is used to provide the executor with the information it needs to perform its task
     */
    @AllArgsConstructor
    private static class UpdateBlockedModelTask {
        private String modelId;
        private boolean isRemoveRequest;
    }

    /**
     * Updates the cluster state based on the UpdateBlockedModelTask
     */
    private static class UpdateBlockedModelExecutor implements ClusterStateTaskExecutor<UpdateBlockedModelTask> {
        /**
         * @param clusterState ClusterState
         * @param taskList contains the list of UpdateBlockedModelTask request parameters (modelId and isRemoveRequest)
         * @return Represents the result of a batched execution of cluster state update tasks (UpdateBlockedModelTasks)
         */
        @Override
        public ClusterTasksResult<UpdateBlockedModelTask> execute(ClusterState clusterState, List<UpdateBlockedModelTask> taskList) {

            // Check if the objects are not null and throw a customized NullPointerException
            Objects.requireNonNull(clusterState, "Cluster state must not be null");
            Objects.requireNonNull(clusterState.metadata(), "Cluster metadata must not be null");
            ModelGraveyard immutableModelGraveyard = clusterState.metadata().custom(ModelGraveyard.TYPE);
            ModelGraveyard modelGraveyard;
            Set<String> copySet;

            if (immutableModelGraveyard == null) {
                modelGraveyard = new ModelGraveyard();
            } else {
                // Deep Copy to copy all the modelIds in ModelGraveyard to local object
                // to avoid copying the reference
                copySet = new HashSet<>(immutableModelGraveyard.getModelGraveyard());
                modelGraveyard = new ModelGraveyard(copySet);
            }

            for (UpdateBlockedModelTask task : taskList) {
                if (task.isRemoveRequest) {
                    modelGraveyard.remove(task.modelId);
                    continue;
                }
                modelGraveyard.add(task.modelId);
            }

            Metadata.Builder metaDataBuilder = Metadata.builder(clusterState.metadata());
            metaDataBuilder.putCustom(ModelGraveyard.TYPE, modelGraveyard);

            ClusterState updatedClusterState = ClusterState.builder(clusterState).metadata(metaDataBuilder).build();
            return new ClusterTasksResult.Builder<UpdateBlockedModelTask>().successes(taskList).build(updatedClusterState);
        }
    }
}
