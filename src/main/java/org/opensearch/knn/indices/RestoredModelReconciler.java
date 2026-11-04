/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.indices;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.logging.log4j.message.ParameterizedMessage;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.ClearScrollRequest;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchScrollRequest;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataRequest;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.sort.SortOrder;
import org.opensearch.snapshots.RestoreContext;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_METADATA_FIELD;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;

/**
 * Fixes the models of a restored model index whose ids can't be deleted or trained again.
 * <p>
 * A snapshot copies the model docs when it copies the model index's shard, but saves the {@code knn-models} metadata
 * when the whole snapshot finishes. A model that trains while the snapshot runs can come back with a doc that says
 * training, with no doc, or with a doc and metadata that disagree, and a model deleted while it runs comes back with a
 * doc and no metadata. {@link #reconcile} marks the models still training failed, and removes the metadata or doc of a
 * model that has only one of them.
 */
@Log4j2
public class RestoredModelReconciler {

    @VisibleForTesting
    static final String RESTORED_WHILE_TRAINING_ERROR = "Training failed to complete as the model index was restored from a snapshot";
    // The default index.max_result_window, the most docs one page can return.
    private static final int MODELS_PER_PAGE = 10_000;
    private static final TimeValue SCROLL_KEEP_ALIVE = TimeValue.timeValueMinutes(1);

    private final ModelDao modelDao;
    private final ClusterService clusterService;
    private final Client client;
    private final int modelsPerPage;

    public RestoredModelReconciler(ModelDao modelDao, ClusterService clusterService, Client client) {
        this(modelDao, clusterService, client, MODELS_PER_PAGE);
    }

    @VisibleForTesting
    RestoredModelReconciler(ModelDao modelDao, ClusterService clusterService, Client client, int modelsPerPage) {
        this.modelDao = modelDao;
        this.clusterService = clusterService;
        this.client = client;
        this.modelsPerPage = modelsPerPage;
    }

    /**
     * Fixes the models of the model index if {@code context} restored it, and does nothing otherwise. It blocks until
     * every model is checked, so it must not run on a transport or cluster state thread.
     *
     * @param context the finished restore
     */
    public void reconcile(RestoreContext context) {
        if (context.restoredIndices().stream().noneMatch(index -> MODEL_INDEX_NAME.equals(index.getName()))) {
            return;
        }
        // Marking a model failed also writes its knn-models entry, so this runs before the models are compared.
        failModelsTrainingOnMissingNodes();
        removeIncompleteModels();
    }

    /**
     * Marks failed every model whose doc says training on a node that isn't in this cluster. No training job runs for
     * such a model, so without this it stays training and DELETE refuses it.
     */
    private void failModelsTrainingOnMissingNodes() {
        final Set<String> nodeIds = new HashSet<>();
        for (DiscoveryNode node : clusterService.state().nodes()) {
            nodeIds.add(node.getEphemeralId());
        }
        final List<SearchHit> hits;
        try {
            hits = searchAll(QueryBuilders.termQuery(MODEL_STATE, ModelState.TRAINING.getName()));
        } catch (Exception e) {
            log.error("Failed to search the restored model index for models in training", e);
            return;
        }
        for (SearchHit hit : hits) {
            try {
                final ModelMetadata modelMetadata = ModelMetadata.getMetadataFromSourceMap(hit.getSourceAsMap());
                if (nodeIds.contains(modelMetadata.getNodeAssignment())) {
                    continue;
                }
                modelMetadata.setState(ModelState.FAILED);
                modelMetadata.setError(RESTORED_WHILE_TRAINING_ERROR);
                final PlainActionFuture<IndexResponse> future = PlainActionFuture.newFuture();
                modelDao.update(new Model(modelMetadata, null, hit.getId()), future);
                future.actionGet();
                log.info("Model {} marked as {} after the model index was restored", hit.getId(), ModelState.FAILED);
            } catch (Exception e) {
                log.error(new ParameterizedMessage("Failed to mark model {} as failed after the model index was restored", hit.getId()), e);
            }
        }
    }

    /**
     * Removes the {@code knn-models} entry of every model that has no doc, and deletes the doc of every model that has no
     * entry. GET and DELETE read the doc, while training and indices read the entry, so a model with only one of them
     * can't be used, deleted or trained again. A doc that says training is left alone, since training writes the doc
     * before its entry.
     */
    private void removeIncompleteModels() {
        final IndexMetadata indexMetadata = clusterService.state().metadata().index(MODEL_INDEX_NAME);
        if (indexMetadata == null) {
            return;
        }
        final Map<String, String> entries = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
        final Set<String> modelIdsWithEntry = entries == null ? Set.of() : entries.keySet();
        final Map<String, Object> docStates = new HashMap<>();
        try {
            for (SearchHit hit : searchAll(QueryBuilders.matchAllQuery(), MODEL_STATE)) {
                docStates.put(hit.getId(), hit.getSourceAsMap().get(MODEL_STATE));
            }
        } catch (Exception e) {
            log.error("Failed to read the model docs of the restored model index", e);
            return;
        }
        for (String modelId : modelIdsWithEntry) {
            if (docStates.containsKey(modelId)) {
                continue;
            }
            try {
                removeMetadata(modelId);
                log.info("Removed the metadata of model {}, which has no doc in the restored model index", modelId);
            } catch (Exception e) {
                log.error(
                    new ParameterizedMessage("Failed to remove the metadata of model {} after the model index was restored", modelId),
                    e
                );
            }
        }
        for (Map.Entry<String, Object> doc : docStates.entrySet()) {
            if (modelIdsWithEntry.contains(doc.getKey()) || ModelState.TRAINING.getName().equals(doc.getValue())) {
                continue;
            }
            try {
                final PlainActionFuture<DeleteModelResponse> future = PlainActionFuture.newFuture();
                modelDao.delete(doc.getKey(), future);
                future.actionGet();
                log.info("Deleted model {}, which has a doc but no metadata in the restored model index", doc.getKey());
            } catch (Exception e) {
                log.error(new ParameterizedMessage("Failed to delete model {} after the model index was restored", doc.getKey()), e);
            }
        }
    }

    /**
     * Returns every model doc that matches {@code query}, read one page at a time.
     *
     * @param query  the docs to return
     * @param fields the source fields to return, or none for the whole doc
     */
    private List<SearchHit> searchAll(QueryBuilder query, String... fields) throws IOException {
        final SearchSourceBuilder source = new SearchSourceBuilder().query(query).sort("_doc", SortOrder.ASC).size(modelsPerPage);
        if (fields.length > 0) {
            source.fetchSource(fields, null);
        }
        final List<SearchHit> hits = new ArrayList<>();
        SearchResponse response = search(new SearchRequest().scroll(SCROLL_KEEP_ALIVE).source(source));
        try {
            while (response.getHits().getHits().length > 0) {
                hits.addAll(Arrays.asList(response.getHits().getHits()));
                response = scroll(response.getScrollId());
            }
        } finally {
            clearScroll(response.getScrollId());
        }
        return hits;
    }

    private SearchResponse search(SearchRequest request) throws IOException {
        final PlainActionFuture<SearchResponse> future = PlainActionFuture.newFuture();
        modelDao.search(request, future);
        return future.actionGet();
    }

    private SearchResponse scroll(String scrollId) {
        final PlainActionFuture<SearchResponse> future = PlainActionFuture.newFuture();
        try (ThreadContext.StoredContext ignored = client.threadPool().getThreadContext().stashContext()) {
            client.searchScroll(new SearchScrollRequest(scrollId).scroll(SCROLL_KEEP_ALIVE), future);
        }
        return future.actionGet();
    }

    private void clearScroll(String scrollId) {
        final ClearScrollRequest request = new ClearScrollRequest();
        request.addScrollId(scrollId);
        try (ThreadContext.StoredContext ignored = client.threadPool().getThreadContext().stashContext()) {
            client.clearScroll(
                request,
                ActionListener.wrap(response -> {}, e -> log.warn("Failed to clear the scroll over the restored model index", e))
            );
        }
    }

    private void removeMetadata(String modelId) {
        final PlainActionFuture<AcknowledgedResponse> future = PlainActionFuture.newFuture();
        try (ThreadContext.StoredContext ignored = client.threadPool().getThreadContext().stashContext()) {
            client.execute(UpdateModelMetadataAction.INSTANCE, new UpdateModelMetadataRequest(modelId, true, null), future);
        }
        future.actionGet();
    }
}
