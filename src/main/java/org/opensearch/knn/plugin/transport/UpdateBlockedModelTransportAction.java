/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
import org.opensearch.knn.plugin.BlockedModelIds;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.PLUGIN_NAME;

/**
 * Transport action used to update blocked modelIds list on the cluster manager node.
 */
public class UpdateBlockedModelTransportAction extends TransportMasterNodeAction<UpdateBlockedModelRequest, AcknowledgedResponse> {

    public static Logger logger = LogManager.getLogger(UpdateBlockedModelTransportAction.class);

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
    private static class UpdateBlockedModelTask {

        private String modelId;
        private boolean isRemoveRequest;

        /**
         * Constructor
         *
         * @param modelId id of model
         * @param isRemoveRequest should this modelId be removed
         */
        UpdateBlockedModelTask(String modelId, boolean isRemoveRequest) {
            this.modelId = modelId;
            this.isRemoveRequest = isRemoveRequest;
        }
    }

    private static class UpdateBlockedModelExecutor implements ClusterStateTaskExecutor<UpdateBlockedModelTask> {
        @Override
        public ClusterTasksResult<UpdateBlockedModelTask> execute(ClusterState clusterState, List<UpdateBlockedModelTask> list) {

            BlockedModelIds immutableBlockedModelIds = clusterState.metadata().custom(BlockedModelIds.TYPE);
            BlockedModelIds blockedModelIds;

            if (immutableBlockedModelIds == null) {
                blockedModelIds = new BlockedModelIds(new ArrayList<>());
            } else {
                blockedModelIds = immutableBlockedModelIds;
            }

            for (UpdateBlockedModelTask task : list) {
                if (task.isRemoveRequest) {
                    blockedModelIds.remove(task.modelId);
                } else {
                    blockedModelIds.add(task.modelId);
                }
            }

            Metadata.Builder metaDataBuilder = Metadata.builder(clusterState.metadata());
            metaDataBuilder.putCustom(BlockedModelIds.TYPE, blockedModelIds);

            ClusterState updatedClusterState = ClusterState.builder(clusterState).metadata(metaDataBuilder).build();
            return new ClusterTasksResult.Builder<UpdateBlockedModelTask>().successes(list).build(updatedClusterState);
        }
    }
}
