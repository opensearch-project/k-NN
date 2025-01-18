/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;

/**
 * Base class for integration tests for KNN plugin. Contains several methods for testing KNN ES functionality.
 */
public class KNNTestCase extends OpenSearchTestCase {

    protected static final KNNLibrarySearchContext EMPTY_ENGINE_SPECIFIC_CONTEXT = ctx -> Map.of();

    @Mock
    protected ClusterService clusterService;
    private AutoCloseable openMocks;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks = MockitoAnnotations.openMocks(this);
        // This is required to make sure that before every test we are initializing the KNNSettings. Not doing this
        // leads to failures of unit tests cases when a unit test is run separately. Try running this test:
        // ./gradlew ':test' --tests "org.opensearch.knn.training.TrainingJobTests.testRun_success" and see it fails
        // but if run along with other tests this test passes.
        initKNNSettings();
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        resetState();
        openMocks.close();
    }

    @Override
    protected boolean enableWarningsCheck() {
        // Disable warnings check to avoid flaky tests, more details at:
        // https://github.com/opensearch-project/k-NN/issues/1392
        return false;
    }

    public void resetState() {
        // Reset all of the counters
        for (KNNCounter knnCounter : KNNCounter.values()) {
            knnCounter.set(0L);
        }
        initKNNSettings();

        // Clean up the cache
        NativeMemoryCacheManager.getInstance().invalidateAll();
        NativeMemoryCacheManager.getInstance().close();
    }

    private void initKNNSettings() {
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        defaultClusterSettings.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(s -> s.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
        KNNSettings.state().setClusterService(clusterService);
    }

    public Map<String, Object> xContentBuilderToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true, xContentBuilder.contentType()).v2();
    }

    public static KNNMethodContext getDefaultKNNMethodContext() {
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        return new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, methodComponentContext);
    }

    public static KNNMethodContext getDefaultKNNMethodContextForModel() {
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_IVF, Collections.emptyMap());
        return new KNNMethodContext(KNNEngine.FAISS, SpaceType.DEFAULT, methodComponentContext);
    }

    public static KNNMethodContext getDefaultByteKNNMethodContext() {
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        return new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, methodComponentContext);
    }

    public static KNNMethodContext getDefaultBinaryKNNMethodContext() {
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        return new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT_BINARY, methodComponentContext);
    }

    public static KNNMappingConfig getMappingConfigForMethodMapping(KNNMethodContext knnMethodContext, int dimension) {
        return new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(knnMethodContext);
            }

            @Override
            public int getDimension() {
                return dimension;
            }
        };
    }

    public static KNNMappingConfig getMappingConfigForFlatMapping(int dimension) {
        return () -> dimension;
    }

    public static KNNMappingConfig getMappingConfigForModelMapping(String modelId, int dimension) {
        return new KNNMappingConfig() {
            @Override
            public Optional<String> getModelId() {
                return Optional.of(modelId);
            }

            @Override
            public int getDimension() {
                return dimension;
            }
        };
    }

    /**
     * Adjust the provided dimension based on {@link VectorDataType} during ingestion.
     * @param dimension int
     * @param vectorDataType {@link VectorDataType}
     * @return int
     */
    protected int adjustDimensionForIndexing(final int dimension, final VectorDataType vectorDataType) {
        return VectorDataType.BINARY == vectorDataType ? dimension * Byte.SIZE : dimension;
    }

    /**
     * Adjust the provided dimension based on {@link VectorDataType} for search.
     *
     * @param dimension int
     * @param vectorDataType {@link VectorDataType}
     * @return int
     */
    protected int adjustDimensionForSearch(final int dimension, final VectorDataType vectorDataType) {
        return VectorDataType.BINARY == vectorDataType ? dimension / Byte.SIZE : dimension;
    }
}
