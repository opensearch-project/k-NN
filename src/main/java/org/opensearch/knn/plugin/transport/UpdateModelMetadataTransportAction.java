/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.transport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.action.support.clustermanager.TransportClusterManagerNodeAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.ClusterStateTaskConfig;
import org.opensearch.cluster.ClusterStateTaskExecutor;
import org.opensearch.cluster.ClusterStateTaskListener;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.Priority;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.PLUGIN_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_METADATA_FIELD;

/**
 * Transport action used to update metadata of model's on the cluster manager node.
 */
public class UpdateModelMetadataTransportAction extends TransportClusterManagerNodeAction<
    UpdateModelMetadataRequest,
    AcknowledgedResponse> {

    public static Logger logger = LogManager.getLogger(UpdateModelMetadataTransportAction.class);

    private UpdateModelMetadataExecutor updateModelMetadataExecutor;

    @Inject
    public UpdateModelMetadataTransportAction(
        TransportService transportService,
        ClusterService clusterService,
        ThreadPool threadPool,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(
            UpdateModelMetadataAction.NAME,
            transportService,
            clusterService,
            threadPool,
            actionFilters,
            UpdateModelMetadataRequest::new,
            indexNameExpressionResolver
        );
        this.updateModelMetadataExecutor = new UpdateModelMetadataExecutor();
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
        UpdateModelMetadataRequest request,
        ClusterState clusterState,
        ActionListener<AcknowledgedResponse> actionListener
    ) {
        // ClusterManager updates model metadata based on request parameters
        clusterService.submitStateUpdateTask(
            PLUGIN_NAME,
            new UpdateModelMetaDataTask(request.getModelId(), request.getModelMetadata(), request.isRemoveRequest()),
            ClusterStateTaskConfig.build(Priority.NORMAL),
            updateModelMetadataExecutor,
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
    protected ClusterBlockException checkBlock(UpdateModelMetadataRequest request, ClusterState clusterState) {
        return null;
    }

    /**
     * UpdateModelMetaDataTask is used to provide the executor with the information it needs to perform its task
     */
    private static class UpdateModelMetaDataTask {

        private String modelId;
        private ModelMetadata modelMetadata;
        private boolean isRemoveRequest;

        /**
         * Constructor
         *
         * @param modelId id of model
         * @param modelMetadata metadata for the model
         * @param isRemoveRequest should this model be removed
         */
        UpdateModelMetaDataTask(String modelId, ModelMetadata modelMetadata, boolean isRemoveRequest) {
            this.modelId = modelId;
            this.modelMetadata = modelMetadata;
            this.isRemoveRequest = isRemoveRequest;
        }
    }

    private static class UpdateModelMetadataExecutor implements ClusterStateTaskExecutor<UpdateModelMetaDataTask> {

        @Override
        public ClusterTasksResult<UpdateModelMetaDataTask> execute(ClusterState clusterState, List<UpdateModelMetaDataTask> list) {
            // Get the map of the models metadata
            IndexMetadata indexMetadata = clusterState.metadata().index(MODEL_INDEX_NAME);

            if (indexMetadata == null) {
                throw new RuntimeException("Model index's metadata does not exist");
            }

            Map<String, String> immutableModels = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
            Map<String, String> models;

            // If the field doesnt exist, we need to create a new map
            if (immutableModels == null) {
                models = new HashMap<>();
            } else {
                models = new HashMap<>(immutableModels);
            }

            // Update the map
            for (UpdateModelMetaDataTask task : list) {
                if (task.isRemoveRequest) {
                    models.remove(task.modelId);
                } else {
                    models.put(task.modelId, task.modelMetadata.toString());
                }
            }

            // Write the map back to the cluster metadata
            Metadata.Builder metaDataBuilder = Metadata.builder(clusterState.metadata());

            metaDataBuilder.put(IndexMetadata.builder(indexMetadata).putCustom(MODEL_METADATA_FIELD, models));

            ClusterState updatedClusterState = ClusterState.builder(clusterState).metadata(metaDataBuilder).build();
            return new ClusterTasksResult.Builder<UpdateModelMetaDataTask>().successes(list).build(updatedClusterState);
        }
    }
}
