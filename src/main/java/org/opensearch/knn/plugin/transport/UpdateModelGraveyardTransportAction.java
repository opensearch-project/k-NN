/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.action.support.clustermanager.TransportClusterManagerNodeAction;
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
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.indices.ModelGraveyard;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.PLUGIN_NAME;

/**
 * Transport action used to update model graveyard on the cluster manager node.
 */
@Log4j2
public class UpdateModelGraveyardTransportAction extends TransportClusterManagerNodeAction<
    UpdateModelGraveyardRequest,
    AcknowledgedResponse> {
    private UpdateModelGraveyardExecutor updateModelGraveyardExecutor;

    @Inject
    public UpdateModelGraveyardTransportAction(
        TransportService transportService,
        ClusterService clusterService,
        ThreadPool threadPool,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(
            UpdateModelGraveyardAction.NAME,
            transportService,
            clusterService,
            threadPool,
            actionFilters,
            UpdateModelGraveyardRequest::new,
            indexNameExpressionResolver
        );
        this.updateModelGraveyardExecutor = new UpdateModelGraveyardExecutor();
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
    protected void clusterManagerOperation(
        UpdateModelGraveyardRequest request,
        ClusterState clusterState,
        ActionListener<AcknowledgedResponse> actionListener
    ) {
        // ClusterManager updates model graveyard based on request parameters
        clusterService.submitStateUpdateTask(
            PLUGIN_NAME,
            new UpdateModelGraveyardTask(request.getModelId(), request.isRemoveRequest()),
            ClusterStateTaskConfig.build(Priority.NORMAL),
            updateModelGraveyardExecutor,
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
    protected ClusterBlockException checkBlock(UpdateModelGraveyardRequest request, ClusterState clusterState) {
        return null;
    }

    /**
     * UpdateModelGraveyardTask is used to provide the executor with the information it needs to perform its task
     */
    @Value
    private static class UpdateModelGraveyardTask {
        String modelId;
        boolean isRemoveRequest;
    }

    /**
     * Updates the cluster state based on the UpdateModelGraveyardTask
     */
    private static class UpdateModelGraveyardExecutor implements ClusterStateTaskExecutor<UpdateModelGraveyardTask> {
        /**
         * @param clusterState ClusterState
         * @param taskList contains the list of UpdateModelGraveyardTask request parameters (modelId and isRemoveRequest)
         * @return Represents the result of a batched execution of cluster state update tasks (UpdateModelGraveyardTasks)
         */
        @Override
        public ClusterTasksResult<UpdateModelGraveyardTask> execute(ClusterState clusterState, List<UpdateModelGraveyardTask> taskList) {

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
                copySet = new HashSet<>(immutableModelGraveyard.getModelIds());
                modelGraveyard = new ModelGraveyard(copySet);
            }

            for (UpdateModelGraveyardTask task : taskList) {
                if (task.isRemoveRequest()) {
                    modelGraveyard.remove(task.getModelId());
                    continue;
                }
                modelGraveyard.add(task.getModelId());
            }

            Metadata.Builder metaDataBuilder = Metadata.builder(clusterState.metadata());
            metaDataBuilder.putCustom(ModelGraveyard.TYPE, modelGraveyard);

            ClusterState updatedClusterState = ClusterState.builder(clusterState).metadata(metaDataBuilder).build();
            return new ClusterTasksResult.Builder<UpdateModelGraveyardTask>().successes(taskList).build(updatedClusterState);
        }
    }
}
