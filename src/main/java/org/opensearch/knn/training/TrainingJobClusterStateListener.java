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

package org.opensearch.knn.training;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.ClusterStateListener;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.search.SearchHit;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;

/**
 * TrainingJobClusterStateListener is a ClusterStateListener that is used to update models that are still training when a node leaves or the cluster crashes.
 * This class also sets a flag in TrainingJobRunner to block serialization when a node rejoins a cluster.
 */
@Log4j2
public class TrainingJobClusterStateListener implements ClusterStateListener {
    private static TrainingJobClusterStateListener INSTANCE;

    private static ModelDao modelDao;
    private static ThreadPool threadPool;
    private static ClusterService clusterService;
    private String oldClusterManagerNodeId = "";
    private String currentClusterManagerNodeId = "";
    private boolean clusterManagerNodeRemoved = false;

    /**
     * Get singleton instance of TrainingJobRunner
     *
     * @return singleton instance of TrainingJobRunner
     */
    public static synchronized TrainingJobClusterStateListener getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new TrainingJobClusterStateListener();
        }
        return INSTANCE;
    }

    /**
     * Initializes static components.
     *
     * @param threadPool threadPool to use to schedule update of models
     * @param modelDao modelDao used to get modelIds
     * @param clusterService clusterService used to add a listener
     */
    public static synchronized void initialize(ThreadPool threadPool, ModelDao modelDao, ClusterService clusterService) {
        TrainingJobClusterStateListener.threadPool = threadPool;
        TrainingJobClusterStateListener.modelDao = modelDao;
        TrainingJobClusterStateListener.clusterService = clusterService;
    }

    /**
     * This method is called whenever the cluster state changes.
     * It is used to update models that are still training when a node leaves or the cluster crashes.
     * It is also used to cancel training jobs when a node rejoins the cluster.
     * @param event the event that changed the cluster change
     */
    @Override
    public void clusterChanged(ClusterChangedEvent event) {
        if (event.localNodeClusterManager()) {
            if (event.isNewCluster()) {
                // When the cluster is first created, the cluster manager will update models that are still marked as training.
                threadPool.schedule(() -> {
                    try {
                        updateModelsNewCluster();
                    } catch (IOException | InterruptedException | ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                }, TimeValue.timeValueSeconds(1), ThreadPool.Names.GENERIC);
            } else if (event.nodesRemoved()) {
                List<DiscoveryNode> removedNodes = event.nodesDelta().removedNodes();
                threadPool.schedule(() -> {
                    try {
                        updateModelsNodesRemoved(removedNodes);
                    } catch (IOException | InterruptedException | ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                }, TimeValue.timeValueSeconds(0), ThreadPool.Names.GENERIC);
            }
        }
    }

    protected void updateModelsNewCluster() throws IOException, InterruptedException, ExecutionException {
        if (modelDao.isCreated()) {
            List<String> modelIds = searchModelIds();
            for (String modelId : modelIds) {
                ModelMetadata modelMetadata = getModelMetadata(modelId);
                if (modelMetadata.getState().equals(ModelState.TRAINING)) {
                    updateModelStateAsFailed(modelId, modelMetadata, "Training failed to complete as cluster crashed");
                }
            }
        }
    }

    protected void updateModelsNodesRemoved(List<DiscoveryNode> removedNodes) throws IOException, InterruptedException, ExecutionException {
        if (modelDao.isCreated()) {
            List<String> modelIds = searchModelIds();
            for (DiscoveryNode removedNode : removedNodes) {
                for (String modelId : modelIds) {
                    ModelMetadata modelMetadata = getModelMetadata(modelId);
                    if (modelMetadata.getNodeAssignment().equals(removedNode.getEphemeralId())
                        && modelMetadata.getState().equals(ModelState.TRAINING)) {
                        updateModelStateAsFailed(modelId, modelMetadata, "Training failed to complete as node dropped");
                    }
                }
            }
        }
    }

    private List<String> searchModelIds() throws IOException, InterruptedException {
        List<String> modelIds = new ArrayList<String>();
        CountDownLatch latch = new CountDownLatch(1);
        modelDao.search(new SearchRequest(), new ActionListener<SearchResponse>() {
            @Override
            public void onResponse(SearchResponse searchResponse) {
                try {
                    for (SearchHit searchHit : searchResponse.getHits().getHits()) {
                        modelIds.add(searchHit.getId());
                    }
                } finally {
                    latch.countDown();
                }
            }

            @Override
            public void onFailure(Exception e) {
                latch.countDown();
            }
        });
        latch.await();
        return modelIds;
    }

    private void updateModelStateAsFailed(String modelId, ModelMetadata modelMetadata, String msg) throws IOException, ExecutionException,
        InterruptedException {
        modelMetadata.setState(ModelState.FAILED);
        modelMetadata.setError(msg);
        Model model = new Model(modelMetadata, null, modelId);
        modelDao.update(model, new ActionListener<IndexResponse>() {
            @Override
            public void onResponse(IndexResponse indexResponse) {
                log.info("Model {} marked as {}", model.getModelID(), model.getModelMetadata().getState());
            }

            @Override
            public void onFailure(Exception e) {
                log.error("Failed to update model state", e);
            }
        });
    }

    private ModelMetadata getModelMetadata(String modelId) throws ExecutionException, InterruptedException {
        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        // On versions prior to 2.14, only models in created state are present in model metadata.
        if (modelMetadata == null) {
            log.info(
                "Model metadata is null in cluster metadata. This can happen for models training on nodes prior to OpenSearch version 2.14.0.  Fetching model information from system index."
            );
            Model model = modelDao.get(modelId);
            return model.getModelMetadata();
        }
        return modelMetadata;
    }
}
