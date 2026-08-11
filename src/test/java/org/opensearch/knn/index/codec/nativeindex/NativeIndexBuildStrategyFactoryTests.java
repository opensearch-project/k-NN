/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class NativeIndexBuildStrategyFactoryTests extends KNNTestCase {

    private FieldInfo fieldInfo;
    private KNNVectorValues<?> knnVectorValues;
    private Supplier<RepositoriesService> repositoriesServiceSupplier;
    private IndexSettings indexSettings;
    private KNNLibraryIndexingContext knnLibraryIndexingContext;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        fieldInfo = mock(FieldInfo.class);
        knnVectorValues = mock(KNNVectorValues.class);
        repositoriesServiceSupplier = mock(Supplier.class);
        indexSettings = mock(IndexSettings.class);
        knnLibraryIndexingContext = mock(KNNLibraryIndexingContext.class);
        // Contexts produced by AbstractKNNMethod always carry a non-null spec; mirror that in the mock
        when(knnLibraryIndexingContext.getResolvedSpec()).thenReturn(
            ResolvedIndexSpec.builder()
                .engine(KNNEngine.FAISS)
                .methodName(METHOD_HNSW)
                .encoderType(Encoder.EncoderType.FLAT)
                .vectorDataType(VectorDataType.FLOAT)
                .dimension(128)
                .indexVersionCreated(Version.CURRENT)
                .build()
        );
    }

    @SneakyThrows
    public void testGetBuildStrategy_faissNonTemplate_returnsMemOptimized() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(false);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();
            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(MemOptimizedNativeIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_templateField_returnsDefault() {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("model_id", "test-model");
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(false);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();
            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(DefaultIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_nonFaissEngine_returnsDefault() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.LUCENE);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(false);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();
            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(DefaultIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_remoteConditionsMet_returnsRemoteStrategy() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class);
            MockedStatic<RemoteIndexBuildStrategy> mockedRemote = Mockito.mockStatic(RemoteIndexBuildStrategy.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(true);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            // totalLiveDocs = 10 > MIN_DOCS_FOR_REMOTE_INDEX_BUILD (4)
            int totalLiveDocs = 10;
            long vectorBlobLength = 32L * totalLiveDocs;

            mockedRemote.when(() -> RemoteIndexBuildStrategy.shouldBuildIndexRemotely(any(IndexSettings.class), anyLong()))
                .thenReturn(true);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory(repositoriesServiceSupplier, indexSettings);
            // FAISS supports remote index build when the context carries a spec that supports it
            ResolvedIndexSpec remoteBuildSpec = ResolvedIndexSpec.builder()
                .engine(KNNEngine.FAISS)
                .methodName(METHOD_HNSW)
                .encoderType(Encoder.EncoderType.FLAT)
                .vectorDataType(VectorDataType.FLOAT)
                .dimension(128)
                .indexVersionCreated(Version.CURRENT)
                .build();
            when(knnLibraryIndexingContext.getResolvedSpec()).thenReturn(remoteBuildSpec);
            factory.setKnnLibraryIndexingContext(knnLibraryIndexingContext);

            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, totalLiveDocs, knnVectorValues);

            assertTrue(strategy instanceof RemoteIndexBuildStrategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_remoteBuildDisabled_returnsLocalStrategy() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(false);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory(repositoriesServiceSupplier, indexSettings);
            factory.setKnnLibraryIndexingContext(knnLibraryIndexingContext);

            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(MemOptimizedNativeIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_tooFewDocs_returnsLocalStrategy() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(true);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            // totalLiveDocs = 3, which is <= MIN_DOCS_FOR_REMOTE_INDEX_BUILD (4)
            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory(repositoriesServiceSupplier, indexSettings);
            factory.setKnnLibraryIndexingContext(knnLibraryIndexingContext);

            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 3, knnVectorValues);

            assertSame(MemOptimizedNativeIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_nullRepositoriesServiceSupplier_returnsLocalStrategy() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(true);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);
            when(indexSettings.getValue(KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)).thenReturn(false);
            when(indexSettings.getIndex()).thenReturn(new org.opensearch.core.index.Index("test-index", "uuid"));

            // null repositoriesServiceSupplier
            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory(null, indexSettings);
            factory.setKnnLibraryIndexingContext(knnLibraryIndexingContext);

            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(MemOptimizedNativeIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_nullIndexSettings_returnsLocalStrategy() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(true);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            // null indexSettings
            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory(repositoriesServiceSupplier, null);
            factory.setKnnLibraryIndexingContext(knnLibraryIndexingContext);

            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(MemOptimizedNativeIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_defaultConstructor_returnsLocalStrategy() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(true);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            // Default constructor sets both repositoriesServiceSupplier and indexSettings to null
            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();

            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(MemOptimizedNativeIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_faissWithTemplate_returnsDefault() {
        // FAISS engine + model_id present => isTemplate=true, iterative=false => DefaultIndexBuildStrategy
        Map<String, String> attributes = new HashMap<>();
        attributes.put("model_id", "some-model");
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.FAISS);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(false);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();
            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(DefaultIndexBuildStrategy.getInstance(), strategy);
        }
    }

    @SneakyThrows
    public void testGetBuildStrategy_nmslibEngine_returnsDefault() {
        Map<String, String> attributes = new HashMap<>();
        when(fieldInfo.attributes()).thenReturn(attributes);

        try (
            MockedStatic<FieldInfoExtractor> mockedExtractor = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<KNNCodecUtil> mockedCodecUtil = Mockito.mockStatic(KNNCodecUtil.class);
            MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)
        ) {
            mockedExtractor.when(() -> FieldInfoExtractor.extractKNNEngine(fieldInfo)).thenReturn(KNNEngine.NMSLIB);
            mockedCodecUtil.when(() -> KNNCodecUtil.initializeVectorValues(any())).thenAnswer(i -> null);
            mockedSettings.when(KNNSettings::isKNNRemoteVectorBuildEnabled).thenReturn(false);
            when(knnVectorValues.bytesPerVector()).thenReturn(32);

            NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();
            NativeIndexBuildStrategy strategy = factory.getBuildStrategy(fieldInfo, 10, knnVectorValues);

            assertSame(DefaultIndexBuildStrategy.getInstance(), strategy);
        }
    }
}
