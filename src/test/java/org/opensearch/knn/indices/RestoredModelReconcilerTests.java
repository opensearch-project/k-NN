/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.indices;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.ClearScrollRequest;
import org.opensearch.action.search.ClearScrollResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchScrollRequest;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.index.Index;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataRequest;
import org.opensearch.knn.training.TrainingJobClusterStateListener;
import org.opensearch.plugins.PluginsService;
import org.opensearch.snapshots.RestoreContext;
import org.opensearch.snapshots.Snapshot;
import org.opensearch.snapshots.SnapshotId;
import org.opensearch.transport.client.Client;

import java.nio.charset.StandardCharsets;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_METADATA_FIELD;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.indices.RestoredModelReconciler.RESTORED_WHILE_TRAINING_ERROR;

public class RestoredModelReconcilerTests extends KNNSingleNodeTestCase {

    // A node id from the cluster that took the snapshot, which is never a node of this cluster.
    private static final String SOURCE_CLUSTER_NODE = "node-of-the-source-cluster";
    private static final byte[] MODEL_BLOB = "model".getBytes(StandardCharsets.UTF_8);

    private static MockedStatic<TrainingJobClusterStateListener> trainingJobClusterStateListenerMockedStatic;

    // The listener fails training models when a cluster starts. Each test starts a new node, so it is mocked out to
    // keep it from changing the models under test.
    @BeforeClass
    public static void setUpListener() {
        trainingJobClusterStateListenerMockedStatic = mockStatic(TrainingJobClusterStateListener.class);
        TrainingJobClusterStateListener trainingJobClusterStateListener = mock(TrainingJobClusterStateListener.class);
        doNothing().when(trainingJobClusterStateListener).clusterChanged(any(ClusterChangedEvent.class));
        trainingJobClusterStateListenerMockedStatic.when(TrainingJobClusterStateListener::getInstance)
            .thenReturn(trainingJobClusterStateListener);
    }

    @AfterClass
    public static void tearDownListener() {
        trainingJobClusterStateListenerMockedStatic.close();
    }

    public void testReconcile_whenModelIndexNotRestored_thenModelsAreUnchanged() throws Exception {
        putModel("training", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        putModel("no-doc", ModelState.CREATED, "");
        deleteDoc("no-doc");
        putModel("no-metadata", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("no-metadata");

        reconciler().reconcile(restoreOf(new Index("another-index", "uuid")));

        assertEquals(ModelState.TRAINING, docState("training"));
        assertTrue(metadata().containsKey("no-doc"));
        assertTrue(hasDoc("no-metadata"));
    }

    public void testReconcile_whenDocAndMetadataSayTrainingOnNodeNotInCluster_thenModelIsFailed() throws Exception {
        putModel("training", ModelState.TRAINING, SOURCE_CLUSTER_NODE);

        reconciler().reconcile(restoreOfModelIndex());

        assertFailedAfterRestore("training");
    }

    public void testReconcile_whenDocSaysTrainingAndMetadataSaysCreated_thenModelIsFailed() throws Exception {
        putModel("trained-during-snapshot", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        setDocToTraining("trained-during-snapshot");
        assertEquals(ModelState.CREATED, modelDao().getMetadata("trained-during-snapshot").getState());

        reconciler().reconcile(restoreOfModelIndex());

        assertFailedAfterRestore("trained-during-snapshot");
    }

    public void testReconcile_whenDocSaysTrainingAndMetadataIsMissing_thenModelIsFailed() throws Exception {
        putModel("doc-only", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        removeMetadata("doc-only");

        reconciler().reconcile(restoreOfModelIndex());

        assertFailedAfterRestore("doc-only");
    }

    public void testReconcile_whenModelTrainsOnNodeInCluster_thenModelStaysTraining() throws Exception {
        putModel("training-here", ModelState.TRAINING, clusterService().localNode().getEphemeralId());

        reconciler().reconcile(restoreOfModelIndex());

        assertEquals(ModelState.TRAINING, docState("training-here"));
        assertEquals(ModelState.TRAINING, modelDao().getMetadata("training-here").getState());
    }

    public void testReconcile_whenModelsAreCreatedOrFailed_thenModelsAreUnchanged() throws Exception {
        putModel("created", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        putModel("failed", ModelState.FAILED, SOURCE_CLUSTER_NODE);
        final Map<String, String> metadataBefore = metadata();

        reconciler().reconcile(restoreOfModelIndex());

        assertEquals(ModelState.CREATED, docState("created"));
        assertEquals(ModelState.FAILED, docState("failed"));
        assertEquals(metadataBefore, metadata());
    }

    public void testReconcile_whenMetadataSaysCreatedAndDocIsMissing_thenMetadataIsRemoved() throws Exception {
        putModel("created-no-doc", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("created-no-doc");

        reconciler().reconcile(restoreOfModelIndex());

        assertFalse(metadata().containsKey("created-no-doc"));
    }

    public void testReconcile_whenMetadataSaysTrainingAndDocIsMissing_thenMetadataIsRemoved() throws Exception {
        putModel("training-no-doc", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        deleteDoc("training-no-doc");

        reconciler().reconcile(restoreOfModelIndex());

        assertFalse(metadata().containsKey("training-no-doc"));
    }

    public void testReconcile_whenCreatedDocHasNoMetadata_thenDocIsDeleted() throws Exception {
        // A model deleted after the snapshot copies the model index's shard.
        putModel("deleted-during-snapshot", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("deleted-during-snapshot");

        reconciler().reconcile(restoreOfModelIndex());

        assertFalse(hasDoc("deleted-during-snapshot"));
        assertFalse(metadata().containsKey("deleted-during-snapshot"));
    }

    public void testReconcile_whenFailedDocHasNoMetadata_thenDocIsDeleted() throws Exception {
        putModel("failed-no-metadata", ModelState.FAILED, SOURCE_CLUSTER_NODE);
        removeMetadata("failed-no-metadata");

        reconciler().reconcile(restoreOfModelIndex());

        assertFalse(hasDoc("failed-no-metadata"));
    }

    public void testReconcile_whenTrainingDocWithoutMetadataIsOnNodeInCluster_thenDocIsKept() throws Exception {
        // Training writes the doc before its knn-models entry, so a model being trained here can briefly look like this.
        putModel("being-trained-here", ModelState.TRAINING, clusterService().localNode().getEphemeralId());
        removeMetadata("being-trained-here");

        reconciler().reconcile(restoreOfModelIndex());

        assertEquals(ModelState.TRAINING, docState("being-trained-here"));
    }

    public void testReconcile_whenDeletingOneDocFails_thenOtherDocsAreDeleted() throws Exception {
        putModel("delete-fails", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("delete-fails");
        putModel("no-metadata", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("no-metadata");
        final ModelDao failingDelete = spy(modelDao());
        doAnswer(invocation -> {
            final ActionListener<DeleteModelResponse> listener = invocation.getArgument(1);
            listener.onFailure(new RuntimeException("delete failed"));
            return null;
        }).when(failingDelete).delete(eq("delete-fails"), any());

        new RestoredModelReconciler(failingDelete, clusterService(), client()).reconcile(restoreOfModelIndex());

        assertTrue(hasDoc("delete-fails"));
        assertFalse(hasDoc("no-metadata"));
    }

    public void testReconcile_whenAnotherModelsMetadataCannotBeParsed_thenStuckModelsAreFixed() throws Exception {
        // A training error with commas, like a faiss error, splits the knn-models entry into too many fields to parse.
        putModel("failed-with-commas", ModelState.FAILED, SOURCE_CLUSTER_NODE, "Error in f(a, b, c): too few vectors");
        final String unparsableEntry = metadata().get("failed-with-commas");
        expectThrows(IllegalArgumentException.class, () -> ModelMetadata.fromString(unparsableEntry));
        putModel("training", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        putModel("no-doc", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("no-doc");
        putModel("no-metadata", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("no-metadata");

        reconciler().reconcile(restoreOfModelIndex());

        assertFailedAfterRestore("training");
        assertFalse(metadata().containsKey("no-doc"));
        assertFalse(hasDoc("no-metadata"));
        assertEquals(unparsableEntry, metadata().get("failed-with-commas"));
    }

    public void testReconcile_whenMoreThanTenModelsAreStuck_thenAllAreFixed() throws Exception {
        final List<String> training = IntStream.range(0, 12).mapToObj(i -> "training-" + i).toList();
        final List<String> noDoc = IntStream.range(0, 12).mapToObj(i -> "no-doc-" + i).toList();
        final List<String> noMetadata = IntStream.range(0, 12).mapToObj(i -> "no-metadata-" + i).toList();
        for (String modelId : training) {
            putModel(modelId, ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        }
        for (String modelId : noDoc) {
            putModel(modelId, ModelState.CREATED, SOURCE_CLUSTER_NODE);
            deleteDoc(modelId);
        }
        for (String modelId : noMetadata) {
            putModel(modelId, ModelState.CREATED, SOURCE_CLUSTER_NODE);
            removeMetadata(modelId);
        }

        reconciler().reconcile(restoreOfModelIndex());

        for (String modelId : training) {
            assertFailedAfterRestore(modelId);
        }
        for (String modelId : noDoc) {
            assertFalse(modelId, metadata().containsKey(modelId));
        }
        for (String modelId : noMetadata) {
            assertFalse(modelId, hasDoc(modelId));
        }
    }

    public void testReconcile_whenModelDocsTakeSeveralPages_thenAllModelsAreChecked() throws Exception {
        final List<String> noDoc = IntStream.range(0, 5).mapToObj(i -> "no-doc-" + i).toList();
        final List<String> noMetadata = IntStream.range(0, 5).mapToObj(i -> "no-metadata-" + i).toList();
        for (String modelId : noDoc) {
            putModel(modelId, ModelState.CREATED, SOURCE_CLUSTER_NODE);
            deleteDoc(modelId);
        }
        for (String modelId : noMetadata) {
            putModel(modelId, ModelState.CREATED, SOURCE_CLUSTER_NODE);
            removeMetadata(modelId);
        }
        putModel("created", ModelState.CREATED, SOURCE_CLUSTER_NODE);

        new RestoredModelReconciler(modelDao(), clusterService(), client(), 2).reconcile(restoreOfModelIndex());

        assertEquals(List.of("created"), List.copyOf(metadata().keySet()));
        for (String modelId : noMetadata) {
            assertFalse(modelId, hasDoc(modelId));
        }
        assertTrue(hasDoc("created"));
    }

    public void testReconcile_whenReadingTheNextPageOfDocsFails_thenIncompleteModelsAreKept() throws Exception {
        putModel("no-doc", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("no-doc");
        putModel("no-metadata", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("no-metadata");
        final Client failingScroll = spy(client());
        doAnswer(invocation -> {
            final ActionListener<SearchResponse> listener = invocation.getArgument(1);
            listener.onFailure(new RuntimeException("scroll failed"));
            return null;
        }).when(failingScroll).searchScroll(any(SearchScrollRequest.class), any());

        new RestoredModelReconciler(modelDao(), clusterService(), failingScroll).reconcile(restoreOfModelIndex());

        assertTrue(metadata().containsKey("no-doc"));
        assertTrue(hasDoc("no-metadata"));
    }

    public void testReconcile_whenClearingTheScrollFails_thenIncompleteModelsAreRemoved() throws Exception {
        putModel("no-doc", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("no-doc");
        putModel("no-metadata", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("no-metadata");
        final Client failingClearScroll = spy(client());
        doAnswer(invocation -> {
            final ActionListener<ClearScrollResponse> listener = invocation.getArgument(1);
            listener.onFailure(new RuntimeException("clear scroll failed"));
            return null;
        }).when(failingClearScroll).clearScroll(any(ClearScrollRequest.class), any());

        new RestoredModelReconciler(modelDao(), clusterService(), failingClearScroll).reconcile(restoreOfModelIndex());

        assertFalse(metadata().containsKey("no-doc"));
        assertFalse(hasDoc("no-metadata"));
    }

    public void testReconcile_whenModelsInTrainingTakeSeveralPages_thenAllAreFailed() throws Exception {
        for (int i = 0; i < 5; i++) {
            putModel("training-" + i, ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        }

        new RestoredModelReconciler(modelDao(), clusterService(), client(), 2).reconcile(restoreOfModelIndex());

        for (int i = 0; i < 5; i++) {
            assertFailedAfterRestore("training-" + i);
        }
    }

    public void testReconcile_whenSearchFails_thenModelsAreUnchanged() throws Exception {
        putModel("training", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        putModel("no-doc", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("no-doc");
        putModel("no-metadata", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        removeMetadata("no-metadata");
        final ModelDao failingSearch = mock(ModelDao.class);
        doAnswer(invocation -> {
            final ActionListener<SearchResponse> listener = invocation.getArgument(1);
            listener.onFailure(new RuntimeException("search failed"));
            return null;
        }).when(failingSearch).search(any(), any());

        new RestoredModelReconciler(failingSearch, clusterService(), client()).reconcile(restoreOfModelIndex());

        assertEquals(ModelState.TRAINING, docState("training"));
        assertTrue(metadata().containsKey("no-doc"));
        assertTrue(hasDoc("no-metadata"));
    }

    public void testReconcile_whenMarkingOneModelFailedFails_thenOtherModelsAreFailed() throws Exception {
        putModel("update-fails", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        putModel("training", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        final ModelDao failingUpdate = spy(modelDao());
        doAnswer(invocation -> {
            final ActionListener<IndexResponse> listener = invocation.getArgument(1);
            listener.onFailure(new RuntimeException("update failed"));
            return null;
        }).when(failingUpdate).update(argThat(model -> "update-fails".equals(model.getModelID())), any());

        new RestoredModelReconciler(failingUpdate, clusterService(), client()).reconcile(restoreOfModelIndex());

        assertEquals(ModelState.TRAINING, docState("update-fails"));
        assertFailedAfterRestore("training");
    }

    public void testReconcile_whenRemovingOneMetadataFails_thenOtherMetadataIsRemoved() throws Exception {
        putModel("remove-fails", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("remove-fails");
        putModel("no-doc", ModelState.CREATED, SOURCE_CLUSTER_NODE);
        deleteDoc("no-doc");
        final Client failingRemove = spy(client());
        doAnswer(invocation -> {
            final ActionListener<AcknowledgedResponse> listener = invocation.getArgument(2);
            listener.onFailure(new RuntimeException("remove failed"));
            return null;
        }).when(failingRemove)
            .execute(
                eq(UpdateModelMetadataAction.INSTANCE),
                argThat((UpdateModelMetadataRequest request) -> "remove-fails".equals(request.getModelId())),
                any()
            );

        new RestoredModelReconciler(modelDao(), clusterService(), failingRemove).reconcile(restoreOfModelIndex());

        assertTrue(metadata().containsKey("remove-fails"));
        assertFalse(metadata().containsKey("no-doc"));
    }

    public void testReconcile_whenModelIndexHasNoModels_thenNothingChanges() {
        createIndex(MODEL_INDEX_NAME);

        reconciler().reconcile(restoreOfModelIndex());

        assertNull(clusterService().state().metadata().index(MODEL_INDEX_NAME).getCustomData(MODEL_METADATA_FIELD));
    }

    public void testReconcile_whenModelIndexIsMissing_thenNothingHappens() {
        reconciler().reconcile(restoreOf(new Index(MODEL_INDEX_NAME, "uuid")));

        assertNull(clusterService().state().metadata().index(MODEL_INDEX_NAME));
    }

    public void testOnRestore_whenModelIndexRestored_thenStuckModelIsFailed() throws Exception {
        putModel("training", ModelState.TRAINING, SOURCE_CLUSTER_NODE);
        final KNNPlugin plugin = getInstanceFromNode(PluginsService.class).filterPlugins(KNNPlugin.class).get(0);

        plugin.onRestore(restoreOfModelIndex());

        assertFailedAfterRestore("training");
    }

    private RestoredModelReconciler reconciler() {
        return new RestoredModelReconciler(modelDao(), clusterService(), client());
    }

    private static ModelDao modelDao() {
        return ModelDao.OpenSearchKNNModelDao.getInstance();
    }

    private ClusterService clusterService() {
        return getInstanceFromNode(ClusterService.class);
    }

    private RestoreContext restoreOfModelIndex() {
        return restoreOf(clusterService().state().metadata().index(MODEL_INDEX_NAME).getIndex());
    }

    private static RestoreContext restoreOf(Index index) {
        return new RestoreContext(new Snapshot("repository", new SnapshotId("snapshot", "uuid")), "restore-uuid", List.of(index), false);
    }

    private void putModel(String modelId, ModelState state, String nodeAssignment) {
        putModel(modelId, state, nodeAssignment, "");
    }

    /** Writes the model doc and its knn-models entry, the way training does. */
    private void putModel(String modelId, ModelState state, String nodeAssignment, String error) {
        final ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.DEFAULT,
            2,
            state,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            error,
            nodeAssignment,
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        final PlainActionFuture<IndexResponse> future = PlainActionFuture.newFuture();
        try {
            modelDao().put(new Model(modelMetadata, state == ModelState.CREATED ? MODEL_BLOB : null, modelId), future);
        } catch (Exception e) {
            throw new AssertionError(e);
        }
        future.actionGet();
    }

    /** Rewrites the model doc as a model still training, leaving its knn-models entry as it is. */
    private void setDocToTraining(String modelId) {
        final Map<String, Object> source = new HashMap<>(client().prepareGet(MODEL_INDEX_NAME, modelId).get().getSourceAsMap());
        source.put(MODEL_STATE, ModelState.TRAINING.getName());
        source.remove(MODEL_BLOB_PARAMETER);
        client().prepareIndex(MODEL_INDEX_NAME)
            .setId(modelId)
            .setSource(source)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE)
            .get();
    }

    /** Deletes the model doc, leaving its knn-models entry as it is. */
    private void deleteDoc(String modelId) {
        client().prepareDelete(MODEL_INDEX_NAME, modelId).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE).get();
    }

    /** Removes the model's knn-models entry, leaving its doc as it is. */
    private void removeMetadata(String modelId) {
        client().execute(UpdateModelMetadataAction.INSTANCE, new UpdateModelMetadataRequest(modelId, true, null)).actionGet();
    }

    private boolean hasDoc(String modelId) {
        return client().prepareGet(MODEL_INDEX_NAME, modelId).get().isExists();
    }

    private ModelState docState(String modelId) {
        final Map<String, Object> source = client().prepareGet(MODEL_INDEX_NAME, modelId).get().getSourceAsMap();
        return ModelState.getModelState((String) source.get(MODEL_STATE));
    }

    /** The knn-models entries, unparsed, by model id. */
    private Map<String, String> metadata() {
        final IndexMetadata indexMetadata = clusterService().state().metadata().index(MODEL_INDEX_NAME);
        final Map<String, String> models = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
        return models == null ? Map.of() : Map.copyOf(models);
    }

    private void assertFailedAfterRestore(String modelId) throws Exception {
        final Model model = modelDao().get(modelId);
        assertEquals(modelId, ModelState.FAILED, model.getModelMetadata().getState());
        assertEquals(modelId, RESTORED_WHILE_TRAINING_ERROR, model.getModelMetadata().getError());
        final ModelMetadata entry = modelDao().getMetadata(modelId);
        assertEquals(modelId, ModelState.FAILED, entry.getState());
        assertEquals(modelId, RESTORED_WHILE_TRAINING_ERROR, entry.getError());
    }
}
