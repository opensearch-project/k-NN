/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */
package org.opensearch.knn.indices;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.admin.indices.create.CreateIndexAction;
import org.opensearch.action.delete.DeleteAction;
import org.opensearch.action.get.GetAction;
import org.opensearch.action.index.IndexAction;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.support.ActionFilter;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.CheckedRunnable;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.env.Environment;
import org.opensearch.env.NodeEnvironment;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheAction;
import org.opensearch.knn.plugin.transport.UpdateModelGraveyardAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.training.TrainingJobClusterStateListener;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.PluginsService;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.script.ScriptService;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;
import org.opensearch.watcher.ResourceWatcherService;

import java.nio.charset.StandardCharsets;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;

/**
 * Tests that every action ModelDao runs, including the k-NN metadata, graveyard and cache actions that
 * follow a model write or delete, runs as the plugin subject and not as the caller, while the caller's
 * listener still runs in the caller's context.
 */
public class ModelDaoPluginSubjectTests extends KNNSingleNodeTestCase {

    private static final String CALLER_HEADER = "knn-test-caller";
    private static final String CALLER = "set-by-caller";
    private static final String NO_CALLER = "no caller header";

    private static MockedStatic<TrainingJobClusterStateListener> trainingJobClusterStateListener;

    @BeforeClass
    public static void silenceTrainingJobSweep() {
        // As in ModelDaoTests: the sweep that a new cluster schedules would read models this test is
        // still writing, from another thread.
        TrainingJobClusterStateListener listener = mock(TrainingJobClusterStateListener.class);
        trainingJobClusterStateListener = mockStatic(TrainingJobClusterStateListener.class);
        trainingJobClusterStateListener.when(TrainingJobClusterStateListener::getInstance).thenReturn(listener);
    }

    @AfterClass
    public static void restoreTrainingJobSweep() {
        trainingJobClusterStateListener.close();
    }

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return List.of(KNNPlugin.class, RecordingFilterPlugin.class);
    }

    public void testModelActionsRunAsThePluginSubjectAndListenersAsTheCaller() throws Exception {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        String modelId = "plugin-subject-model";

        // The model index does not exist yet, so put creates it first.
        PlainActionFuture<String> putListener = new PlainActionFuture<>();
        asCaller(() -> modelDao.put(createdModel(modelId), recordingCaller(putListener)));
        assertEquals("the put listener must run in the caller's context", CALLER, putListener.get(10, TimeUnit.SECONDS));
        assertRanOnlyAsThePluginSubject(
            CreateIndexAction.NAME,
            IndexAction.NAME,
            UpdateModelMetadataAction.NAME,
            RemoveModelFromCacheAction.NAME
        );

        PlainActionFuture<String> updateListener = new PlainActionFuture<>();
        asCaller(() -> modelDao.update(createdModel(modelId), recordingCaller(updateListener)));
        assertEquals("the update listener must run in the caller's context", CALLER, updateListener.get(10, TimeUnit.SECONDS));
        assertRanOnlyAsThePluginSubject(IndexAction.NAME, UpdateModelMetadataAction.NAME, RemoveModelFromCacheAction.NAME);

        // The blocking get that ModelCache loads models through.
        asCaller(() -> {
            assertEquals(modelId, modelDao.get(modelId).getModelID());
            assertEquals("the calling thread must keep the caller's context", CALLER, callerHeader());
        });
        assertRanOnlyAsThePluginSubject(GetAction.NAME);

        PlainActionFuture<String> searchListener = new PlainActionFuture<>();
        asCaller(() -> modelDao.search(new SearchRequest(), recordingCaller(searchListener)));
        assertEquals("the search listener must run in the caller's context", CALLER, searchListener.get(10, TimeUnit.SECONDS));
        assertRanOnlyAsThePluginSubject(SearchAction.NAME);

        PlainActionFuture<String> deleteListener = new PlainActionFuture<>();
        asCaller(() -> modelDao.delete(modelId, recordingCaller(deleteListener)));
        assertEquals("the delete listener must run in the caller's context", CALLER, deleteListener.get(10, TimeUnit.SECONDS));
        assertRanOnlyAsThePluginSubject(
            GetAction.NAME,
            UpdateModelGraveyardAction.NAME,
            UpdateModelMetadataAction.NAME,
            DeleteAction.NAME,
            RemoveModelFromCacheAction.NAME
        );
    }

    public void testRecorderSeesTheCallerOnAnActionTheCallerRuns() throws Exception {
        // Shows the filter records the caller header when it is there, so the absence asserted above means something.
        PlainActionFuture<String> listenerCaller = new PlainActionFuture<>();
        asCaller(() -> client().admin().indices().prepareCreate("caller-index").execute(recordingCaller(listenerCaller)));
        assertEquals(CALLER, listenerCaller.get(10, TimeUnit.SECONDS));

        assertEquals(List.of(CALLER), recorder().seen.get(CreateIndexAction.NAME));
    }

    /**
     * Asserts that these actions ran since the last check and that nothing, these or the internal actions they
     * trigger, ran with the caller's context. Then resets the record.
     */
    private void assertRanOnlyAsThePluginSubject(String... actions) {
        Map<String, List<String>> seen = recorder().seen;
        for (String action : actions) {
            assertTrue(action + " must have run, saw " + seen.keySet(), seen.containsKey(action));
        }
        seen.forEach(
            (action, callers) -> assertTrue(
                action + " must not run with the caller's context, saw " + callers,
                callers.stream().allMatch(NO_CALLER::equals)
            )
        );
        seen.clear();
    }

    private RecordingFilterPlugin recorder() {
        return getInstanceFromNode(PluginsService.class).filterPlugins(RecordingFilterPlugin.class).get(0);
    }

    private String callerHeader() {
        return Optional.ofNullable(client().threadPool().getThreadContext().getHeader(CALLER_HEADER)).orElse(NO_CALLER);
    }

    private void asCaller(CheckedRunnable<Exception> body) throws Exception {
        ThreadContext threadContext = client().threadPool().getThreadContext();
        try (ThreadContext.StoredContext ignored = threadContext.stashContext()) {
            threadContext.putHeader(CALLER_HEADER, CALLER);
            body.run();
        }
    }

    /** Completes the future with the caller header the listener saw, or fails it with the action's failure. */
    private <T> ActionListener<T> recordingCaller(PlainActionFuture<String> future) {
        return ActionListener.wrap(response -> future.onResponse(callerHeader()), future::onFailure);
    }

    private static Model createdModel(String modelId) {
        return new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                2,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT,
                Mode.NOT_CONFIGURED,
                CompressionLevel.NOT_CONFIGURED,
                Version.CURRENT
            ),
            "hello".getBytes(StandardCharsets.UTF_8),
            modelId
        );
    }

    /** Records, for each action, the caller header in the thread context the action ran with. */
    public static class RecordingFilterPlugin extends Plugin implements ActionPlugin {

        final Map<String, List<String>> seen = new ConcurrentHashMap<>();

        private volatile ThreadContext threadContext;

        /** Keeps the node's thread context for the filter to read. */
        @Override
        public Collection<Object> createComponents(
            Client client,
            ClusterService clusterService,
            ThreadPool threadPool,
            ResourceWatcherService resourceWatcherService,
            ScriptService scriptService,
            NamedXContentRegistry xContentRegistry,
            Environment environment,
            NodeEnvironment nodeEnvironment,
            NamedWriteableRegistry namedWriteableRegistry,
            IndexNameExpressionResolver indexNameExpressionResolver,
            Supplier<RepositoriesService> repositoriesServiceSupplier
        ) {
            threadContext = threadPool.getThreadContext();
            return List.of();
        }

        /** Returns a filter that records each action before letting it run. */
        @Override
        public List<ActionFilter> getActionFilters() {
            Consumer<String> record = action -> seen.computeIfAbsent(action, a -> new CopyOnWriteArrayList<>())
                .add(Optional.ofNullable(threadContext.getHeader(CALLER_HEADER)).orElse(NO_CALLER));
            return List.of(new ActionFilter.Simple() {
                @Override
                public int order() {
                    return Integer.MIN_VALUE;
                }

                @Override
                protected boolean apply(String action, ActionRequest request, ActionListener<?> listener) {
                    record.accept(action);
                    return true;
                }
            });
        }
    }
}
