/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.settings.Settings;
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
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.common.exception.DeleteModelException;
import org.opensearch.knn.indices.ModelGraveyard;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.*;

import static org.opensearch.knn.common.KNNConstants.*;

/**
 * Transport action used to update model graveyard on the cluster manager node.
 */
@Log4j2
public class UpdateModelGraveyardTransportAction extends TransportClusterManagerNodeAction<
    UpdateModelGraveyardRequest,
    AcknowledgedResponse> {
    private UpdateModelGraveyardExecutor updateModelGraveyardExecutor;
    private final IndicesService indicesService;

    @Inject
    public UpdateModelGraveyardTransportAction(
        TransportService transportService,
        ClusterService clusterService,
        ThreadPool threadPool,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver,
        IndicesService indicesService
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
        this.indicesService = indicesService;
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
            new UpdateModelGraveyardTask(request.getModelId(), request.isRemoveRequest(), indicesService),
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
        IndicesService indicesService;
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
        public ClusterTasksResult<UpdateModelGraveyardTask> execute(ClusterState clusterState, List<UpdateModelGraveyardTask> taskList)
            throws IOException {

            // Check if the objects are not null and throw a customized NullPointerException
            Objects.requireNonNull(clusterState, "Cluster state must not be null");
            Objects.requireNonNull(clusterState.metadata(), "Cluster metadata must not be null");
            ModelGraveyard immutableModelGraveyard = clusterState.metadata().custom(ModelGraveyard.TYPE);
            ModelGraveyard modelGraveyard;
            Set<String> copySet;

            Set<String> indicesSet = clusterState.metadata().indices().keySet();

            String[] indicesArray = Arrays.copyOf(indicesSet.toArray(), indicesSet.size(), String[].class);

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
                checkIfIndicesAreUsingModel(clusterState, task, indicesArray);
                modelGraveyard.add(task.getModelId());
            }

            Metadata.Builder metaDataBuilder = Metadata.builder(clusterState.metadata());
            metaDataBuilder.putCustom(ModelGraveyard.TYPE, modelGraveyard);

            ClusterState updatedClusterState = ClusterState.builder(clusterState).metadata(metaDataBuilder).build();
            return new ClusterTasksResult.Builder<UpdateModelGraveyardTask>().successes(taskList).build(updatedClusterState);
        }
    }

    private static void checkIfIndicesAreUsingModel(ClusterState clusterState, UpdateModelGraveyardTask task, String[] indicesArray)
        throws IOException {
        Map<String, MappingMetadata> mappings = clusterState.metadata()
            .findMappings(indicesArray, task.getIndicesService().getFieldFilter());
        Map<String, IndexMetadata> indices = clusterState.metadata().indices();
        List<String> indicesUsingModel = new ArrayList<>();
        // Parse indices and add to list if using the model
        for (Map.Entry<String, MappingMetadata> entry : mappings.entrySet()) {
            MappingMetadata mappingMetadata = entry.getValue();
            Settings indexSettings = indices.get(entry.getKey()).getSettings();
            if (mappingMetadata != null && Boolean.parseBoolean(indexSettings.get("index.knn", "false"))) {
                indicesUsingModel = parseMappingMetadata(mappingMetadata, task, indicesUsingModel, entry.getKey());
            }
        }
        // Throw exception if any indices are using the model
        if (!indicesUsingModel.isEmpty()) {
            throw new DeleteModelException(
                String.format(
                    "Cannot delete model [%s].  Model is in use by the following indices %s, which must be deleted first.",
                    task.getModelId(),
                    indicesUsingModel
                )
            );
        }
    }

    private static List<String> parseMappingMetadata(
        MappingMetadata mappingMetadata,
        UpdateModelGraveyardTask task,
        List<String> indicesUsingModel,
        String index
    ) {
        Map<String, Object> mappingMetadataSourceMap = mappingMetadata.getSourceAsMap();
        String modelIdMappingString = String.join("model_id=", task.getModelId());
        // If modelId is present, parse map to field.
        if (mappingMetadataSourceMap.toString().contains(modelIdMappingString)) {
            for (Map.Entry<String, Object> sourceEntry : mappingMetadataSourceMap.entrySet()) {
                if (sourceEntry.getKey() != null && sourceEntry.getKey().equals(PROPERTIES) && sourceEntry.getValue() instanceof Map) {
                    Map<String, Object> fieldsMap = (Map<String, Object>) sourceEntry.getValue();
                    indicesUsingModel = parseFieldsMap(fieldsMap, task, indicesUsingModel, index);
                }
            }
        }
        return indicesUsingModel;
    }

    private static List<String> parseFieldsMap(
        Map<String, Object> fieldsMap,
        UpdateModelGraveyardTask task,
        List<String> indicesUsingModel,
        String index
    ) {
        for (Map.Entry<String, Object> fieldsEntry : fieldsMap.entrySet()) {
            if (fieldsEntry.getKey() != null && fieldsEntry.getValue() instanceof Map) {
                Map<String, Object> innerMap = (Map<String, Object>) fieldsEntry.getValue();
                for (Map.Entry<String, Object> innerEntry : innerMap.entrySet()) {
                    // If model is in use, fail delete model request
                    if (innerEntry.getKey().equals(MODEL_ID)
                        && innerEntry.getValue() instanceof String
                        && innerEntry.getValue().equals(task.getModelId())) {
                        indicesUsingModel.add(index);
                    }
                }
            }
        }
        return indicesUsingModel;
    }
}
