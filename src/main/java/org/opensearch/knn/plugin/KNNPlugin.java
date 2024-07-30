/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.cluster.NamedDiff;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.core.ParseField;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.index.codec.CodecServiceFactory;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.indices.SystemIndexDescriptor;
import org.opensearch.knn.index.KNNCircuitBreaker;
import org.opensearch.knn.index.KNNClusterUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.codec.KNNCodecService;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.indices.ModelGraveyard;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.plugin.rest.RestDeleteModelHandler;
import org.opensearch.knn.plugin.rest.RestGetModelHandler;
import org.opensearch.knn.plugin.rest.RestKNNStatsHandler;
import org.opensearch.knn.plugin.rest.RestKNNWarmupHandler;
import org.opensearch.knn.plugin.rest.RestSearchModelHandler;
import org.opensearch.knn.plugin.rest.RestTrainModelHandler;
import org.opensearch.knn.plugin.rest.RestClearCacheHandler;
import org.opensearch.knn.plugin.script.KNNScoringScriptEngine;
import org.opensearch.knn.plugin.stats.KNNStats;
import org.opensearch.knn.plugin.transport.DeleteModelAction;
import org.opensearch.knn.plugin.transport.DeleteModelTransportAction;
import org.opensearch.knn.plugin.transport.GetModelAction;
import org.opensearch.knn.plugin.transport.GetModelTransportAction;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsTransportAction;
import org.opensearch.knn.plugin.transport.KNNWarmupAction;
import org.opensearch.knn.plugin.transport.KNNWarmupTransportAction;
import org.opensearch.knn.plugin.transport.ClearCacheAction;
import org.opensearch.knn.plugin.transport.ClearCacheTransportAction;
import com.google.common.collect.ImmutableList;

import org.opensearch.action.ActionRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.settings.SettingsFilter;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.env.Environment;
import org.opensearch.env.NodeEnvironment;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheAction;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheTransportAction;
import org.opensearch.knn.plugin.transport.SearchModelAction;
import org.opensearch.knn.plugin.transport.SearchModelTransportAction;
import org.opensearch.knn.plugin.transport.TrainingJobRouteDecisionInfoAction;
import org.opensearch.knn.plugin.transport.TrainingJobRouteDecisionInfoTransportAction;
import org.opensearch.knn.plugin.transport.TrainingJobRouterAction;
import org.opensearch.knn.plugin.transport.TrainingJobRouterTransportAction;
import org.opensearch.knn.plugin.transport.TrainingModelAction;
import org.opensearch.knn.plugin.transport.TrainingModelRequest;
import org.opensearch.knn.plugin.transport.TrainingModelTransportAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataTransportAction;
import org.opensearch.knn.plugin.transport.UpdateModelGraveyardAction;
import org.opensearch.knn.plugin.transport.UpdateModelGraveyardTransportAction;
import org.opensearch.knn.training.TrainingJobClusterStateListener;
import org.opensearch.knn.training.TrainingJobRunner;
import org.opensearch.knn.training.VectorReader;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.EnginePlugin;
import org.opensearch.plugins.ExtensiblePlugin;
import org.opensearch.plugins.MapperPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.ScriptPlugin;
import org.opensearch.plugins.SearchPlugin;
import org.opensearch.plugins.SystemIndexPlugin;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestHandler;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.opensearch.script.ScriptService;
import org.opensearch.threadpool.ExecutorBuilder;
import org.opensearch.threadpool.FixedExecutorBuilder;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.watcher.ResourceWatcherService;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

import static java.util.Collections.singletonList;
import static org.opensearch.knn.common.KNNConstants.KNN_THREAD_POOL_PREFIX;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

/**
 * Entry point for the KNN plugin where we define mapper for knn_vector type
 * and new query clause "knn"
 *
 *
 * Example Mapping for knn_vector type
 * "settings" : {
 *    "index": {
 *       "knn": true
 *     }
 *   },
 * "mappings": {
 *   "properties": {
 *     "my_vector": {
 *       "type": "knn_vector",
 *       "dimension": 4
 *     }
 *   }
 * }
 *
 * Example Query
 *
 *   "knn": {
 *    "my_vector": {
 *      "vector": [3, 4],
 *      "k": 3
 *    }
 *   }
 *
 */
public class KNNPlugin extends Plugin
    implements
        MapperPlugin,
        SearchPlugin,
        ActionPlugin,
        EnginePlugin,
        ScriptPlugin,
        ExtensiblePlugin,
        SystemIndexPlugin {

    public static final String LEGACY_KNN_BASE_URI = "/_opendistro/_knn";
    public static final String KNN_BASE_URI = "/_plugins/_knn";

    private KNNStats knnStats;
    private ClusterService clusterService;

    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        return Collections.singletonMap(
            KNNVectorFieldMapper.CONTENT_TYPE,
            new KNNVectorFieldMapper.TypeParser(ModelDao.OpenSearchKNNModelDao::getInstance)
        );
    }

    @Override
    public List<QuerySpec<?>> getQueries() {
        return singletonList(new QuerySpec<>(KNNQueryBuilder.NAME, KNNQueryBuilder::new, KNNQueryBuilderParser::fromXContent));
    }

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
        this.clusterService = clusterService;

        // Initialize Native Memory loading strategies
        NativeMemoryLoadStrategy.IndexLoadStrategy.initialize(resourceWatcherService);
        VectorReader vectorReader = new VectorReader(client);
        NativeMemoryLoadStrategy.TrainingLoadStrategy.initialize(vectorReader);

        KNNSettings.state().initialize(client, clusterService);
        KNNClusterUtil.instance().initialize(clusterService);
        ModelDao.OpenSearchKNNModelDao.initialize(client, clusterService, environment.settings());
        ModelCache.initialize(ModelDao.OpenSearchKNNModelDao.getInstance(), clusterService);
        TrainingJobRunner.initialize(threadPool, ModelDao.OpenSearchKNNModelDao.getInstance());
        TrainingJobClusterStateListener.initialize(threadPool, ModelDao.OpenSearchKNNModelDao.getInstance(), clusterService);
        KNNCircuitBreaker.getInstance().initialize(threadPool, clusterService, client);
        KNNQueryBuilder.initialize(ModelDao.OpenSearchKNNModelDao.getInstance());
        KNNVectorFieldMapperUtil.initialize(ModelDao.OpenSearchKNNModelDao.getInstance());
        KNNWeight.initialize(ModelDao.OpenSearchKNNModelDao.getInstance());
        TrainingModelRequest.initialize(ModelDao.OpenSearchKNNModelDao.getInstance(), clusterService);

        clusterService.addListener(TrainingJobClusterStateListener.getInstance());

        knnStats = new KNNStats();
        return ImmutableList.of(knnStats);
    }

    @Override
    public List<Setting<?>> getSettings() {
        return KNNSettings.state().getSettings();
    }

    public List<RestHandler> getRestHandlers(
        Settings settings,
        RestController restController,
        ClusterSettings clusterSettings,
        IndexScopedSettings indexScopedSettings,
        SettingsFilter settingsFilter,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<DiscoveryNodes> nodesInCluster
    ) {

        RestKNNStatsHandler restKNNStatsHandler = new RestKNNStatsHandler();
        RestKNNWarmupHandler restKNNWarmupHandler = new RestKNNWarmupHandler(
            settings,
            restController,
            clusterService,
            indexNameExpressionResolver
        );
        RestGetModelHandler restGetModelHandler = new RestGetModelHandler();
        RestDeleteModelHandler restDeleteModelHandler = new RestDeleteModelHandler();
        RestTrainModelHandler restTrainModelHandler = new RestTrainModelHandler();
        RestSearchModelHandler restSearchModelHandler = new RestSearchModelHandler();
        RestClearCacheHandler restClearCacheHandler = new RestClearCacheHandler(clusterService, indexNameExpressionResolver);

        return ImmutableList.of(
            restKNNStatsHandler,
            restKNNWarmupHandler,
            restGetModelHandler,
            restDeleteModelHandler,
            restTrainModelHandler,
            restSearchModelHandler,
            restClearCacheHandler
        );
    }

    /**
     * Register action and handler so that transportClient can find proxy for action
     */
    @Override
    public List<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> getActions() {
        return Arrays.asList(
            new ActionHandler<>(KNNStatsAction.INSTANCE, KNNStatsTransportAction.class),
            new ActionHandler<>(KNNWarmupAction.INSTANCE, KNNWarmupTransportAction.class),
            new ActionHandler<>(UpdateModelMetadataAction.INSTANCE, UpdateModelMetadataTransportAction.class),
            new ActionHandler<>(TrainingJobRouteDecisionInfoAction.INSTANCE, TrainingJobRouteDecisionInfoTransportAction.class),
            new ActionHandler<>(GetModelAction.INSTANCE, GetModelTransportAction.class),
            new ActionHandler<>(DeleteModelAction.INSTANCE, DeleteModelTransportAction.class),
            new ActionHandler<>(TrainingJobRouterAction.INSTANCE, TrainingJobRouterTransportAction.class),
            new ActionHandler<>(TrainingModelAction.INSTANCE, TrainingModelTransportAction.class),
            new ActionHandler<>(RemoveModelFromCacheAction.INSTANCE, RemoveModelFromCacheTransportAction.class),
            new ActionHandler<>(SearchModelAction.INSTANCE, SearchModelTransportAction.class),
            new ActionHandler<>(UpdateModelGraveyardAction.INSTANCE, UpdateModelGraveyardTransportAction.class),
            new ActionHandler<>(ClearCacheAction.INSTANCE, ClearCacheTransportAction.class)
        );
    }

    @Override
    public Optional<EngineFactory> getEngineFactory(IndexSettings indexSettings) {
        return Optional.empty();
    }

    @Override
    public Optional<CodecServiceFactory> getCustomCodecServiceFactory(IndexSettings indexSettings) {
        if (indexSettings.getValue(KNNSettings.IS_KNN_INDEX_SETTING)) {
            return Optional.of(KNNCodecService::new);
        }
        return Optional.empty();
    }

    @Override
    public void onIndexModule(IndexModule indexModule) {
        KNNSettings.state().onIndexModule(indexModule);
    }

    /**
     * Sample knn custom script
     *
     * {
     *   "query": {
     *     "script_score": {
     *       "query": {
     *         "match_all": {
     *           "boost": 1
     *         }
     *       },
     *       "script": {
     *         "source": "knn_score",
     *         "lang": "knn",
     *         "params": {
     *           "field": "my_dense_vector",
     *           "vector": [
     *             1,
     *             1
     *           ]
     *         }
     *       }
     *     }
     *   }
     * }
     *
     */
    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return new KNNScoringScriptEngine();
    }

    @Override
    public List<ExecutorBuilder<?>> getExecutorBuilders(Settings settings) {
        return ImmutableList.of(new FixedExecutorBuilder(settings, TRAIN_THREAD_POOL, 1, 1, KNN_THREAD_POOL_PREFIX, false));
    }

    @Override
    public List<NamedWriteableRegistry.Entry> getNamedWriteables() {
        List<NamedWriteableRegistry.Entry> entries = new ArrayList<>();

        entries.add(new NamedWriteableRegistry.Entry(Metadata.Custom.class, ModelGraveyard.TYPE, ModelGraveyard::new));
        entries.add(new NamedWriteableRegistry.Entry(NamedDiff.class, ModelGraveyard.TYPE, ModelGraveyard::readDiffFrom));
        return entries;
    }

    @Override
    public List<NamedXContentRegistry.Entry> getNamedXContent() {
        List<NamedXContentRegistry.Entry> entries = new ArrayList<>();

        entries.add(
            new NamedXContentRegistry.Entry(Metadata.Custom.class, new ParseField(ModelGraveyard.TYPE), ModelGraveyard::fromXContent)
        );
        return entries;
    }

    @Override
    public Collection<SystemIndexDescriptor> getSystemIndexDescriptors(Settings settings) {
        return ImmutableList.of(new SystemIndexDescriptor(MODEL_INDEX_NAME, "Index for storing models used for k-NN indices"));
    }
}
