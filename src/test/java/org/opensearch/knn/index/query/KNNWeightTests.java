/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.collect.Comparators;
import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.mockito.MockedConstruction;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.common.io.PathUtils;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.QuantizationConfigKNNCollector;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.util.KNNVectorAsArraySerializer;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.util.Collections.emptyMap;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class KNNWeightTests extends KNNTestCase {
    private static final String FIELD_NAME = "target_field";
    private static final float[] QUERY_VECTOR = new float[] { 1.8f, 2.4f };
    private static final byte[] BYTE_QUERY_VECTOR = new byte[] { 1, 2 };
    private static final String SEGMENT_NAME = "0";
    private static final int K = 5;
    private static final Set<String> SEGMENT_FILES_NMSLIB = Set.of("_0.cfe", "_0_2011_target_field.hnswc");
    private static final Set<String> SEGMENT_FILES_FAISS = Set.of("_0.cfe", "_0_2011_target_field.faissc");
    private static final Set<String> SEGMENT_MULTI_FIELD_FILES_FAISS = Set.of(
        "_0.cfe",
        "_0_2011_target_field.faissc",
        "_0_2011_long_target_field.faissc"
    );
    private static final String CIRCUIT_BREAKER_LIMIT_100KB = "100Kb";
    private static final Integer EF_SEARCH = 10;
    private static final Map<String, ?> HNSW_METHOD_PARAMETERS = Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH);

    private static final Map<Integer, Float> DOC_ID_TO_SCORES = Map.of(10, 0.4f, 101, 0.05f, 100, 0.8f, 50, 0.52f);
    private static final Map<Integer, Float> FILTERED_DOC_ID_TO_SCORES = Map.of(101, 0.05f, 100, 0.8f, 50, 0.52f);
    private static final Map<Integer, Float> EXACT_SEARCH_DOC_ID_TO_SCORES = Map.of(0, 0.12048191f);
    private static final Map<Integer, Float> BINARY_EXACT_SEARCH_DOC_ID_TO_SCORES = Map.of(0, 0.5f);

    private static final Query FILTER_QUERY = new TermQuery(new Term("foo", "fooValue"));

    private static MockedStatic<NativeMemoryCacheManager> nativeMemoryCacheManagerMockedStatic;
    private static MockedStatic<JNIService> jniServiceMockedStatic;

    private static MockedStatic<KNNSettings> knnSettingsMockedStatic;

    @BeforeClass
    public static void setUpClass() throws Exception {
        final KNNSettings knnSettings = mock(KNNSettings.class);
        knnSettingsMockedStatic = mockStatic(KNNSettings.class);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED))).thenReturn(true);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT))).thenReturn(CIRCUIT_BREAKER_LIMIT_100KB);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_CACHE_ITEM_EXPIRY_ENABLED))).thenReturn(false);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES))).thenReturn(TimeValue.timeValueMinutes(10));

        final ByteSizeValue v = ByteSizeValue.parseBytesSizeValue(
            CIRCUIT_BREAKER_LIMIT_100KB,
            KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT
        );
        knnSettingsMockedStatic.when(KNNSettings::getCircuitBreakerLimit).thenReturn(v);
        knnSettingsMockedStatic.when(KNNSettings::state).thenReturn(knnSettings);
        knnSettingsMockedStatic.when(KNNSettings::isKNNPluginEnabled).thenReturn(true);

        nativeMemoryCacheManagerMockedStatic = mockStatic(NativeMemoryCacheManager.class);

        final NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);
        final NativeMemoryAllocation nativeMemoryAllocation = mock(NativeMemoryAllocation.class);
        when(nativeMemoryCacheManager.get(any(), anyBoolean())).thenReturn(nativeMemoryAllocation);

        nativeMemoryCacheManagerMockedStatic.when(NativeMemoryCacheManager::getInstance).thenReturn(nativeMemoryCacheManager);

        final MockedStatic<PathUtils> pathUtilsMockedStatic = mockStatic(PathUtils.class);
        final Path indexPath = mock(Path.class);
        when(indexPath.toString()).thenReturn("/mydrive/myfolder");
        pathUtilsMockedStatic.when(() -> PathUtils.get(anyString(), anyString())).thenReturn(indexPath);
    }

    @Before
    public void setupBeforeTest() {
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(0);
        jniServiceMockedStatic = mockStatic(JNIService.class);
    }

    @After
    public void tearDownAfterTest() {
        jniServiceMockedStatic.close();
    }

    @SneakyThrows
    public void testQueryResultScoreNmslib() {
        for (SpaceType space : List.of(SpaceType.L2, SpaceType.L1, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT, SpaceType.LINF)) {
            testQueryScore(space::scoreTranslation, SEGMENT_FILES_NMSLIB, Map.of(SPACE_TYPE, space.getValue()));
        }
    }

    @SneakyThrows
    public void testQueryResultScoreFaiss() {
        testQueryScore(
            SpaceType.L2::scoreTranslation,
            SEGMENT_FILES_FAISS,
            Map.of(
                SPACE_TYPE,
                SpaceType.L2.getValue(),
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                PARAMETERS,
                String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
            )
        );
        // score translation for Faiss and inner product is different from default defined in Space enum
        testQueryScore(
            rawScore -> SpaceType.INNER_PRODUCT.scoreTranslation(-1 * rawScore),
            SEGMENT_FILES_FAISS,
            Map.of(
                SPACE_TYPE,
                SpaceType.INNER_PRODUCT.getValue(),
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                PARAMETERS,
                String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
            )
        );

        // multi field
        testQueryScore(
            rawScore -> SpaceType.INNER_PRODUCT.scoreTranslation(-1 * rawScore),
            SEGMENT_MULTI_FIELD_FILES_FAISS,
            Map.of(
                SPACE_TYPE,
                SpaceType.INNER_PRODUCT.getValue(),
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                PARAMETERS,
                String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
            )
        );
    }

    @SneakyThrows
    public void testQueryScoreForFaissWithModel() {
        SpaceType spaceType = SpaceType.L2;
        final Function<Float, Float> scoreTranslator = spaceType::scoreTranslation;
        final String modelId = "modelId";
        jniServiceMockedStatic.when(() -> JNIService.queryIndex(anyLong(), any(), eq(K), isNull(), any(), any(), anyInt(), any()))
            .thenReturn(getKNNQueryResults());

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, (BitSetProducer) null);

        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(modelMetadata.getSpaceType()).thenReturn(spaceType);
        when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
        when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.DEFAULT);
        when(modelMetadata.getMethodComponentContext()).thenReturn(new MethodComponentContext("ivf", emptyMap()));
        when(modelDao.getMetadata(eq("modelId"))).thenReturn(modelMetadata);

        KNNWeight.initialize(modelDao);
        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FSDirectory directory = mock(FSDirectory.class);
        when(reader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            true,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(SEGMENT_FILES_FAISS);
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        final Path path = mock(Path.class);
        when(directory.getDirectory()).thenReturn(path);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(Map.of());
        when(fieldInfo.getAttribute(eq(MODEL_ID))).thenReturn(modelId);

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        final List<Integer> actualDocIds = new ArrayList();
        final Map<Integer, Float> translatedScores = getTranslatedScores(scoreTranslator);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            assertEquals(translatedScores.get(docId) * boost, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testQueryScoreForFaissWithNonExistingModel() throws IOException {
        SpaceType spaceType = SpaceType.L2;
        final String modelId = "modelId";

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, (BitSetProducer) null);

        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(modelMetadata.getSpaceType()).thenReturn(spaceType);

        KNNWeight.initialize(modelDao);
        final KNNWeight knnWeight = new KNNWeight(query, 0.0f);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FSDirectory directory = mock(FSDirectory.class);
        when(reader.directory()).thenReturn(directory);

        final Path path = mock(Path.class);
        when(directory.getDirectory()).thenReturn(path);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(Map.of());
        when(fieldInfo.getAttribute(eq(MODEL_ID))).thenReturn(modelId);

        RuntimeException ex = expectThrows(RuntimeException.class, () -> knnWeight.scorer(leafReaderContext));
        assertEquals(String.format("Model \"%s\" is not created.", modelId), ex.getMessage());
    }

    @SneakyThrows
    public void testShardWithoutFiles() {
        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, (BitSetProducer) null);
        final KNNWeight knnWeight = new KNNWeight(query, 0.0f);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FSDirectory directory = mock(FSDirectory.class);
        when(reader.directory()).thenReturn(directory);

        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            false,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(Set.of());
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        final Path path = mock(Path.class);
        when(directory.getDirectory()).thenReturn(path);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);

        final Scorer knnScorer = knnWeight.scorer(leafReaderContext);
        assertEquals(KNNScorer.emptyScorer(knnWeight), knnScorer);
    }

    @SneakyThrows
    public void testEmptyQueryResults() {
        final KNNQueryResult[] knnQueryResults = new KNNQueryResult[] {};
        jniServiceMockedStatic.when(() -> JNIService.queryIndex(anyLong(), any(), eq(K), isNull(), any(), any(), anyInt(), any()))
            .thenReturn(knnQueryResults);

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, (BitSetProducer) null);
        final KNNWeight knnWeight = new KNNWeight(query, 0.0f);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FSDirectory directory = mock(FSDirectory.class);
        when(reader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            true,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(SEGMENT_FILES_NMSLIB);
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        final Path path = mock(Path.class);
        when(directory.getDirectory()).thenReturn(path);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);

        final Scorer knnScorer = knnWeight.scorer(leafReaderContext);
        assertEquals(KNNScorer.emptyScorer(knnWeight), knnScorer);
    }

    @SneakyThrows
    public void testScorer_whenNoFilterBinary_thenSuccess() {
        validateScorer_whenNoFilter_thenSuccess(true);
    }

    @SneakyThrows
    public void testScorer_whenNoFilter_thenSuccess() {
        validateScorer_whenNoFilter_thenSuccess(false);
    }

    private void validateScorer_whenNoFilter_thenSuccess(final boolean isBinary) throws IOException {
        // Given
        int k = 3;
        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any())
        ).thenReturn(getFilteredKNNQueryResults());

        jniServiceMockedStatic.when(
            () -> JNIService.queryBinaryIndex(
                anyLong(),
                eq(BYTE_QUERY_VECTOR),
                eq(k),
                eq(HNSW_METHOD_PARAMETERS),
                any(),
                any(),
                anyInt(),
                any()
            )
        ).thenReturn(getFilteredKNNQueryResults());
        final SegmentReader reader = mockSegmentReader();
        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final KNNQuery query = isBinary
            ? KNNQuery.builder()
                .field(FIELD_NAME)
                .byteQueryVector(BYTE_QUERY_VECTOR)
                .k(k)
                .indexName(INDEX_NAME)
                .filterQuery(FILTER_QUERY)
                .methodParameters(HNSW_METHOD_PARAMETERS)
                .vectorDataType(VectorDataType.BINARY)
                .build()
            : KNNQuery.builder()
                .field(FIELD_NAME)
                .queryVector(QUERY_VECTOR)
                .k(k)
                .indexName(INDEX_NAME)
                .filterQuery(FILTER_QUERY)
                .methodParameters(HNSW_METHOD_PARAMETERS)
                .vectorDataType(VectorDataType.FLOAT)
                .build();

        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(attributesMap);

        // When
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Then
        assertNotNull(knnScorer);
        if (isBinary) {
            jniServiceMockedStatic.verify(
                () -> JNIService.queryBinaryIndex(
                    anyLong(),
                    eq(BYTE_QUERY_VECTOR),
                    eq(k),
                    eq(HNSW_METHOD_PARAMETERS),
                    any(),
                    any(),
                    anyInt(),
                    any()
                ),
                times(1)
            );
        } else {
            jniServiceMockedStatic.verify(
                () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
                times(1)
            );
        }
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenDoingANN_thenSuccess() {
        validateANNWithFilterQuery_whenDoingANN_thenSuccess(false);
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenDoingANNBinary_thenSuccess() {
        validateANNWithFilterQuery_whenDoingANN_thenSuccess(true);
    }

    public void validateANNWithFilterQuery_whenDoingANN_thenSuccess(final boolean isBinary) throws IOException {
        // Given
        int k = 3;
        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        FixedBitSet filterBitSet = new FixedBitSet(filterDocIds.length);
        for (int docId : filterDocIds) {
            filterBitSet.set(docId);
        }
        if (isBinary) {
            jniServiceMockedStatic.when(
                () -> JNIService.queryBinaryIndex(
                    anyLong(),
                    eq(BYTE_QUERY_VECTOR),
                    eq(k),
                    eq(HNSW_METHOD_PARAMETERS),
                    any(),
                    eq(filterBitSet.getBits()),
                    anyInt(),
                    any()
                )
            ).thenReturn(getFilteredKNNQueryResults());
        } else {
            jniServiceMockedStatic.when(
                () -> JNIService.queryIndex(
                    anyLong(),
                    eq(QUERY_VECTOR),
                    eq(k),
                    eq(HNSW_METHOD_PARAMETERS),
                    any(),
                    eq(filterBitSet.getBits()),
                    anyInt(),
                    any()
                )
            ).thenReturn(getFilteredKNNQueryResults());
        }

        final Bits liveDocsBits = mock(Bits.class);
        for (int filterDocId : filterDocIds) {
            when(liveDocsBits.get(filterDocId)).thenReturn(true);
        }
        when(liveDocsBits.length()).thenReturn(1000);

        final SegmentReader reader = mockSegmentReader();
        when(reader.maxDoc()).thenReturn(filterDocIds.length);
        when(reader.getLiveDocs()).thenReturn(liveDocsBits);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final KNNQuery query = isBinary
            ? KNNQuery.builder()
                .field(FIELD_NAME)
                .byteQueryVector(BYTE_QUERY_VECTOR)
                .vectorDataType(VectorDataType.BINARY)
                .k(k)
                .indexName(INDEX_NAME)
                .filterQuery(FILTER_QUERY)
                .methodParameters(HNSW_METHOD_PARAMETERS)
                .build()
            : KNNQuery.builder()
                .field(FIELD_NAME)
                .queryVector(QUERY_VECTOR)
                .k(k)
                .indexName(INDEX_NAME)
                .filterQuery(FILTER_QUERY)
                .methodParameters(HNSW_METHOD_PARAMETERS)
                .build();

        final Weight filterQueryWeight = mock(Weight.class);
        final Scorer filterScorer = mock(Scorer.class);
        when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);
        // Just to make sure that we are not hitting the exact search condition
        when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(filterDocIds.length + 1));

        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost, filterQueryWeight);

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            isBinary ? SpaceType.HAMMING.getValue() : SpaceType.L2.getValue()
        );

        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(attributesMap);

        // When
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Then
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(FILTERED_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        if (isBinary) {
            jniServiceMockedStatic.verify(
                () -> JNIService.queryBinaryIndex(
                    anyLong(),
                    eq(BYTE_QUERY_VECTOR),
                    eq(k),
                    eq(HNSW_METHOD_PARAMETERS),
                    any(),
                    any(),
                    anyInt(),
                    any()
                ),
                times(1)
            );
        } else {
            jniServiceMockedStatic.verify(
                () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
                times(1)
            );
        }

        final List<Integer> actualDocIds = new ArrayList<>();
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            assertEquals(translatedScores.get(docId) * boost, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    private SegmentReader mockSegmentReader() {
        Path path = mock(Path.class);

        FSDirectory directory = mock(FSDirectory.class);
        when(directory.getDirectory()).thenReturn(path);

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            true,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(SEGMENT_FILES_FAISS);
        SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);

        SegmentReader reader = mock(SegmentReader.class);
        when(reader.directory()).thenReturn(directory);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);
        return reader;
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenExactSearch_thenSuccess() {
        validateANNWithFilterQuery_whenExactSearch_thenSuccess(false);
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenExactSearchBinary_thenSuccess() {
        validateANNWithFilterQuery_whenExactSearch_thenSuccess(true);
    }

    public void validateANNWithFilterQuery_whenExactSearch_thenSuccess(final boolean isBinary) throws IOException {
        try (MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            KNNWeight.initialize(null);
            float[] vector = new float[] { 0.1f, 0.3f };
            byte[] byteVector = new byte[] { 1, 3 };
            int filterDocId = 0;
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            final KNNQuery query = isBinary
                ? new KNNQuery(FIELD_NAME, BYTE_QUERY_VECTOR, K, INDEX_NAME, FILTER_QUERY, null, VectorDataType.BINARY, null)
                : new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, FILTER_QUERY, null, null);
            final Weight filterQueryWeight = mock(Weight.class);
            final Scorer filterScorer = mock(Scorer.class);
            when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);
            // scorer will return 2 documents
            when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(1));
            when(reader.maxDoc()).thenReturn(1);
            final Bits liveDocsBits = mock(Bits.class);
            when(reader.getLiveDocs()).thenReturn(liveDocsBits);
            when(liveDocsBits.get(filterDocId)).thenReturn(true);

            final float boost = (float) randomDoubleBetween(0, 10, true);
            final KNNWeight knnWeight = new KNNWeight(query, boost, filterQueryWeight);
            final Map<String, String> attributesMap = ImmutableMap.of(
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                SPACE_TYPE,
                isBinary ? SpaceType.HAMMING.getValue() : SpaceType.L2.getValue()
            );
            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            final KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);
            final KNNBinaryVectorValues binaryVectorValues = mock(KNNBinaryVectorValues.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(attributesMap);
            if (isBinary) {
                when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.HAMMING.getValue());
            } else {
                when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
            }
            when(fieldInfo.getName()).thenReturn(FIELD_NAME);

            if (isBinary) {
                valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                    .thenReturn(binaryVectorValues);
                when(binaryVectorValues.advance(filterDocId)).thenReturn(filterDocId);
                Mockito.when(binaryVectorValues.getVector()).thenReturn(byteVector);
            } else {
                valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                    .thenReturn(floatVectorValues);
                when(floatVectorValues.advance(filterDocId)).thenReturn(filterDocId);
                Mockito.when(floatVectorValues.getVector()).thenReturn(vector);
            }

            final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
            assertNotNull(knnScorer);
            final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
            assertNotNull(docIdSetIterator);
            assertEquals(1, docIdSetIterator.cost());

            final List<Integer> actualDocIds = new ArrayList<>();
            for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
                actualDocIds.add(docId);
                if (isBinary) {
                    assertEquals(BINARY_EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
                } else {
                    assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
                }
            }
            assertEquals(docIdSetIterator.cost(), actualDocIds.size());
            assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
        }
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenExactSearchAndThresholdComputations_thenSuccess() {
        ModelDao modelDao = mock(ModelDao.class);
        KNNWeight.initialize(modelDao);
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(-1);
        float[] vector = new float[] { 0.1f, 0.3f };
        int filterDocId = 0;
        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, FILTER_QUERY, null, null);
        final Weight filterQueryWeight = mock(Weight.class);
        final Scorer filterScorer = mock(Scorer.class);
        when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);
        // scorer will return 2 documents
        when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(1));
        when(reader.maxDoc()).thenReturn(1);
        final Bits liveDocsBits = mock(Bits.class);
        when(reader.getLiveDocs()).thenReturn(liveDocsBits);
        when(liveDocsBits.get(filterDocId)).thenReturn(true);

        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost, filterQueryWeight);
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.name(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final BinaryDocValues binaryDocValues = mock(BinaryDocValues.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(attributesMap);
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.name());
        when(fieldInfo.getName()).thenReturn(FIELD_NAME);
        when(reader.getBinaryDocValues(FIELD_NAME)).thenReturn(binaryDocValues);
        when(binaryDocValues.advance(filterDocId)).thenReturn(filterDocId);
        BytesRef vectorByteRef = new BytesRef(new KNNVectorAsArraySerializer().floatToByteArray(vector));
        when(binaryDocValues.binaryValue()).thenReturn(vectorByteRef);

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    /**
     * This test ensure that we do the exact search when threshold settings are correct and not using filteredIds<=K
     * condition to do exact search.
     * FilteredIdThreshold: 10
     * FilteredIdThresholdPct: 10%
     * FilteredIdsCount: 6
     * liveDocs : null, as there is no deleted documents
     * MaxDoc: 100
     * K : 1
     */
    @SneakyThrows
    public void testANNWithFilterQuery_whenExactSearchViaThresholdSetting_thenSuccess() {
        ModelDao modelDao = mock(ModelDao.class);
        KNNWeight.initialize(modelDao);
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(10);
        float[] vector = new float[] { 0.1f, 0.3f };
        int k = 1;
        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);
        when(reader.maxDoc()).thenReturn(100);
        when(reader.getLiveDocs()).thenReturn(null);
        final Weight filterQueryWeight = mock(Weight.class);
        final Scorer filterScorer = mock(Scorer.class);
        when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);

        when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(filterDocIds.length));

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, k, INDEX_NAME, FILTER_QUERY, null, null);

        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost, filterQueryWeight);
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.name(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final BinaryDocValues binaryDocValues = mock(BinaryDocValues.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(attributesMap);
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.name());
        when(fieldInfo.getName()).thenReturn(FIELD_NAME);
        when(reader.getBinaryDocValues(FIELD_NAME)).thenReturn(binaryDocValues);
        when(binaryDocValues.advance(0)).thenReturn(0);
        BytesRef vectorByteRef = new BytesRef(new KNNVectorAsArraySerializer().floatToByteArray(vector));
        when(binaryDocValues.binaryValue()).thenReturn(vectorByteRef);

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    /**
     * This test ensure that we do the exact search when threshold settings are correct and not using filteredIds<=K
     * condition to do exact search on binary index
     * FilteredIdThreshold: 10
     * FilteredIdThresholdPct: 10%
     * FilteredIdsCount: 6
     * liveDocs : null, as there is no deleted documents
     * MaxDoc: 100
     * K : 1
     */
    @SneakyThrows
    public void testANNWithFilterQuery_whenExactSearchViaThresholdSettingOnBinaryIndex_thenSuccess() {
        try (MockedStatic<KNNVectorValuesFactory> vectorValuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            KNNWeight.initialize(null);
            knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(10);
            byte[] vector = new byte[] { 1, 3 };
            int k = 1;
            final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };

            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            when(reader.maxDoc()).thenReturn(100);
            when(reader.getLiveDocs()).thenReturn(null);
            final Weight filterQueryWeight = mock(Weight.class);
            final Scorer filterScorer = mock(Scorer.class);
            when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);

            when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(filterDocIds.length));

            final KNNQuery query = new KNNQuery(
                FIELD_NAME,
                BYTE_QUERY_VECTOR,
                k,
                INDEX_NAME,
                FILTER_QUERY,
                null,
                VectorDataType.BINARY,
                null
            );

            final float boost = (float) randomDoubleBetween(0, 10, true);
            final KNNWeight knnWeight = new KNNWeight(query, boost, filterQueryWeight);
            final Map<String, String> attributesMap = ImmutableMap.of(
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                SPACE_TYPE,
                SpaceType.HAMMING.name(),
                PARAMETERS,
                String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "BHNSW32")
            );
            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(attributesMap);
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.HAMMING.getValue());
            when(fieldInfo.getName()).thenReturn(FIELD_NAME);

            KNNBinaryVectorValues knnBinaryVectorValues = mock(KNNBinaryVectorValues.class);

            vectorValuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                .thenReturn(knnBinaryVectorValues);
            when(knnBinaryVectorValues.advance(0)).thenReturn(0);
            when(knnBinaryVectorValues.getVector()).thenReturn(vector);

            final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
            assertNotNull(knnScorer);
            final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
            assertNotNull(docIdSetIterator);
            assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

            final List<Integer> actualDocIds = new ArrayList<>();
            for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
                actualDocIds.add(docId);
                assertEquals(BINARY_EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
            }
            assertEquals(docIdSetIterator.cost(), actualDocIds.size());
            assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
        }
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenEmptyFilterIds_thenReturnEarly() {
        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final Weight filterQueryWeight = mock(Weight.class);
        final Scorer filterScorer = mock(Scorer.class);
        when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);
        when(filterScorer.iterator()).thenReturn(DocIdSetIterator.empty());

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, FILTER_QUERY, null, null);
        final KNNWeight knnWeight = new KNNWeight(query, 0.0f, filterQueryWeight);

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);

        final Scorer knnScorer = knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(0, docIdSetIterator.cost());
        assertEquals(0, docIdSetIterator.cost());
    }

    @SneakyThrows
    public void testANNWithParentsFilter_whenExactSearch_thenSuccess() {
        ModelDao modelDao = mock(ModelDao.class);
        KNNWeight.initialize(modelDao);
        SegmentReader reader = getMockedSegmentReader();

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        // We will have 0, 1 for filteredIds and 2 will be the parent id for both of them
        final Scorer filterScorer = mock(Scorer.class);
        when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(2));
        when(reader.maxDoc()).thenReturn(2);

        // Query vector is {1.8f, 2.4f}, therefore, second vector {1.9f, 2.5f} should be returned in a result
        final List<float[]> vectors = Arrays.asList(new float[] { 0.1f, 0.3f }, new float[] { 1.9f, 2.5f });
        final List<BytesRef> byteRefs = vectors.stream()
            .map(vector -> new BytesRef(new KNNVectorAsArraySerializer().floatToByteArray(vector)))
            .collect(Collectors.toList());
        final BinaryDocValues binaryDocValues = mock(BinaryDocValues.class);
        when(binaryDocValues.binaryValue()).thenReturn(byteRefs.get(0), byteRefs.get(1));
        when(binaryDocValues.advance(anyInt())).thenReturn(0, 1);
        when(reader.getBinaryDocValues(FIELD_NAME)).thenReturn(binaryDocValues);

        // Parent ID 2 in bitset is 100 which is 4
        FixedBitSet parentIds = new FixedBitSet(new long[] { 4 }, 3);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        when(parentFilter.getBitSet(leafReaderContext)).thenReturn(parentIds);

        final Weight filterQueryWeight = mock(Weight.class);
        when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, FILTER_QUERY, parentFilter, null);
        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost, filterQueryWeight);

        // Execute
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Verify
        final List<Float> expectedScores = vectors.stream()
            .map(vector -> SpaceType.L2.getKnnVectorSimilarityFunction().compare(QUERY_VECTOR, vector))
            .collect(Collectors.toList());
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertEquals(1, docIdSetIterator.nextDoc());
        assertEquals(expectedScores.get(1) * boost, knnScorer.score(), 0.01f);
        assertEquals(NO_MORE_DOCS, docIdSetIterator.nextDoc());
    }

    @SneakyThrows
    public void testANNWithParentsFilter_whenDoingANN_thenBitSetIsPassedToJNI() {
        SegmentReader reader = getMockedSegmentReader();
        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        // Prepare parentFilter
        final int[] parentsFilter = { 10, 64 };
        final FixedBitSet bitset = new FixedBitSet(65);
        Arrays.stream(parentsFilter).forEach(i -> bitset.set(i));
        final BitSetProducer bitSetProducer = mock(BitSetProducer.class);

        // Prepare query and weight
        when(bitSetProducer.getBitSet(leafReaderContext)).thenReturn(bitset);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(1)
            .indexName(INDEX_NAME)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .parentsFilter(bitSetProducer)
            .build();

        final KNNWeight knnWeight = new KNNWeight(query, 0.0f);

        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(
                anyLong(),
                eq(QUERY_VECTOR),
                eq(1),
                eq(HNSW_METHOD_PARAMETERS),
                any(),
                any(),
                anyInt(),
                eq(parentsFilter)
            )
        ).thenReturn(getKNNQueryResults());

        // Execute
        Scorer knnScorer = knnWeight.scorer(leafReaderContext);

        // Verify
        jniServiceMockedStatic.verify(
            () -> JNIService.queryIndex(
                anyLong(),
                eq(QUERY_VECTOR),
                eq(1),
                eq(HNSW_METHOD_PARAMETERS),
                any(),
                any(),
                anyInt(),
                eq(parentsFilter)
            )
        );
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());
    }

    @SneakyThrows
    public void testDoANNSearch_whenRadialIsDefined_thenCallJniRadiusQueryIndex() {
        final float[] queryVector = new float[] { 0.1f, 0.3f };
        final float radius = 0.5f;
        final int maxResults = 1000;
        jniServiceMockedStatic.when(
            () -> JNIService.radiusQueryIndex(
                anyLong(),
                eq(queryVector),
                eq(radius),
                eq(HNSW_METHOD_PARAMETERS),
                any(),
                eq(maxResults),
                any(),
                anyInt(),
                any()
            )
        ).thenReturn(getKNNQueryResults());
        KNNQuery.Context context = mock(KNNQuery.Context.class);
        when(context.getMaxResultWindow()).thenReturn(maxResults);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(queryVector)
            .radius(radius)
            .indexName(INDEX_NAME)
            .context(context)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .build();
        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FSDirectory directory = mock(FSDirectory.class);
        when(reader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            true,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(SEGMENT_FILES_FAISS);
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        final Path path = mock(Path.class);
        when(directory.getDirectory()).thenReturn(path);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(
            Map.of(
                SPACE_TYPE,
                SpaceType.L2.getValue(),
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                PARAMETERS,
                String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
            )
        );

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        jniServiceMockedStatic.verify(
            () -> JNIService.radiusQueryIndex(
                anyLong(),
                eq(queryVector),
                eq(radius),
                eq(HNSW_METHOD_PARAMETERS),
                any(),
                eq(maxResults),
                any(),
                anyInt(),
                any()
            )
        );

        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();

        final List<Integer> actualDocIds = new ArrayList<>();
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            assertEquals(translatedScores.get(docId) * boost, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    private SegmentReader getMockedSegmentReader() {
        final SegmentReader reader = mock(SegmentReader.class);
        when(reader.maxDoc()).thenReturn(1);

        // Prepare live docs
        when(reader.getLiveDocs()).thenReturn(null);

        // Prepare directory
        final Path path = mock(Path.class);
        final FSDirectory directory = mock(FSDirectory.class);
        when(directory.getDirectory()).thenReturn(path);
        when(reader.directory()).thenReturn(directory);

        // Prepare segment
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            true,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(SEGMENT_FILES_FAISS);
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        // Prepare fieldInfo
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.name(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.attributes()).thenReturn(attributesMap);
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.name());
        when(fieldInfo.getName()).thenReturn(FIELD_NAME);

        // Prepare fieldInfos
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);

        return reader;
    }

    private void testQueryScore(
        final Function<Float, Float> scoreTranslator,
        final Set<String> segmentFiles,
        final Map<String, String> fileAttributes
    ) throws IOException {
        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(K), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any())
        ).thenReturn(getKNNQueryResults());

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(K)
            .indexName(INDEX_NAME)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .build();
        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new KNNWeight(query, boost);

        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FSDirectory directory = mock(FSDirectory.class);
        when(reader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            true,
            false,
            KNNCodecVersion.current().getDefaultCodecDelegate(),
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(segmentFiles);
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        final Path path = mock(Path.class);
        when(directory.getDirectory()).thenReturn(path);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(fileAttributes);

        String engineName = fieldInfo.attributes().getOrDefault(KNN_ENGINE, KNNEngine.NMSLIB.getName());
        KNNEngine knnEngine = KNNEngine.getEngine(engineName);
        List<String> engineFiles = knnWeight.getEngineFiles(reader, knnEngine.getExtension());
        String expectIndexPath = String.format("%s_%s_%s%s%s", SEGMENT_NAME, 2011, FIELD_NAME, knnEngine.getExtension(), "c");
        assertEquals(engineFiles.get(0), expectIndexPath);

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        final List<Integer> actualDocIds = new ArrayList();
        final Map<Integer, Float> translatedScores = getTranslatedScores(scoreTranslator);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            assertEquals(translatedScores.get(docId) * boost, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    private Map<Integer, Float> getTranslatedScores(Function<Float, Float> scoreTranslator) {
        return DOC_ID_TO_SCORES.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getKey, entry -> scoreTranslator.apply(entry.getValue())));
    }

    private KNNQueryResult[] getKNNQueryResults() {
        return DOC_ID_TO_SCORES.entrySet()
            .stream()
            .map(entry -> new KNNQueryResult(entry.getKey(), entry.getValue()))
            .collect(Collectors.toList())
            .toArray(new KNNQueryResult[0]);
    }

    private KNNQueryResult[] getFilteredKNNQueryResults() {
        return FILTERED_DOC_ID_TO_SCORES.entrySet()
            .stream()
            .map(entry -> new KNNQueryResult(entry.getKey(), entry.getValue()))
            .collect(Collectors.toList())
            .toArray(new KNNQueryResult[0]);
    }

    @SneakyThrows
    public void testANNWithQuantizationParams_whenStateNotFound_thenFail() {
        try (MockedStatic<QuantizationService> quantizationServiceMockedStatic = Mockito.mockStatic(QuantizationService.class)) {
            QuantizationService quantizationService = Mockito.mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);
            QuantizationParams quantizationParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
            Mockito.when(quantizationService.getQuantizationParams(any(FieldInfo.class))).thenReturn(quantizationParams);

            // Given
            int k = 3;
            jniServiceMockedStatic.when(
                () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any())
            ).thenReturn(getFilteredKNNQueryResults());

            jniServiceMockedStatic.when(
                () -> JNIService.queryBinaryIndex(
                    anyLong(),
                    eq(BYTE_QUERY_VECTOR),
                    eq(k),
                    eq(HNSW_METHOD_PARAMETERS),
                    any(),
                    any(),
                    anyInt(),
                    any()
                )
            ).thenReturn(getFilteredKNNQueryResults());
            final SegmentReader reader = mockSegmentReader();
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            final KNNQuery query = KNNQuery.builder()
                .field(FIELD_NAME)
                .queryVector(QUERY_VECTOR)
                .k(k)
                .indexName(INDEX_NAME)
                .filterQuery(FILTER_QUERY)
                .methodParameters(HNSW_METHOD_PARAMETERS)
                .vectorDataType(VectorDataType.FLOAT)
                .build();

            final float boost = (float) randomDoubleBetween(0, 10, true);
            final KNNWeight knnWeight = new KNNWeight(query, boost);
            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            final Map<String, String> attributesMap = ImmutableMap.of(
                KNN_ENGINE,
                KNNEngine.FAISS.getName(),
                PARAMETERS,
                String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
            );

            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(attributesMap);
            // fieldName, new float[0], tempCollector, null)
            doNothing().when(reader).searchNearestVectors(any(), eq(new float[0]), any(), any());

            expectThrows(IllegalStateException.class, () -> knnWeight.scorer(leafReaderContext));
        }
    }

    @SneakyThrows
    public void testANNWithQuantizationParams_thenSuccess() {
        try (MockedStatic<QuantizationService> quantizationServiceMockedStatic = Mockito.mockStatic(QuantizationService.class)) {
            QuantizationService quantizationService = Mockito.mock(QuantizationService.class);
            ScalarQuantizationParams quantizationParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
            Mockito.when(quantizationService.getQuantizationParams(any(FieldInfo.class))).thenReturn(quantizationParams);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);

            float[] meanThresholds = new float[] { 1.2f, 2.3f, 3.4f, 4.5f };
            QuantizationState quantizationState = new OneBitScalarQuantizationState(quantizationParams, meanThresholds);

            try (
                MockedConstruction<QuantizationConfigKNNCollector> quantizationCollectorMockedConstruction = Mockito.mockConstruction(
                    QuantizationConfigKNNCollector.class,
                    (mock, context) -> Mockito.when(mock.getQuantizationState()).thenReturn(quantizationState)
                )
            ) {

                // Given
                int k = 3;
                jniServiceMockedStatic.when(
                    () -> JNIService.queryIndex(
                        anyLong(),
                        eq(QUERY_VECTOR),
                        eq(k),
                        eq(HNSW_METHOD_PARAMETERS),
                        any(),
                        any(),
                        anyInt(),
                        any()
                    )
                ).thenReturn(getFilteredKNNQueryResults());

                jniServiceMockedStatic.when(
                    () -> JNIService.queryBinaryIndex(
                        anyLong(),
                        eq(BYTE_QUERY_VECTOR),
                        eq(k),
                        eq(HNSW_METHOD_PARAMETERS),
                        any(),
                        any(),
                        anyInt(),
                        any()
                    )
                ).thenReturn(getFilteredKNNQueryResults());
                final SegmentReader reader = mockSegmentReader();
                final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
                when(leafReaderContext.reader()).thenReturn(reader);

                final KNNQuery query = KNNQuery.builder()
                    .field(FIELD_NAME)
                    .queryVector(QUERY_VECTOR)
                    .k(k)
                    .indexName(INDEX_NAME)
                    .filterQuery(FILTER_QUERY)
                    .methodParameters(HNSW_METHOD_PARAMETERS)
                    .vectorDataType(VectorDataType.FLOAT)
                    .build();

                final float boost = (float) randomDoubleBetween(0, 10, true);
                final KNNWeight knnWeight = new KNNWeight(query, boost);
                final FieldInfos fieldInfos = mock(FieldInfos.class);
                final FieldInfo fieldInfo = mock(FieldInfo.class);
                final Map<String, String> attributesMap = ImmutableMap.of(
                    KNN_ENGINE,
                    KNNEngine.FAISS.getName(),
                    PARAMETERS,
                    String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
                );

                when(reader.getFieldInfos()).thenReturn(fieldInfos);
                when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
                when(fieldInfo.attributes()).thenReturn(attributesMap);

                KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

                assertNotNull(knnScorer);
                jniServiceMockedStatic.verify(
                    () -> JNIService.queryIndex(
                        anyLong(),
                        eq(QUERY_VECTOR),
                        eq(k),
                        eq(HNSW_METHOD_PARAMETERS),
                        any(),
                        any(),
                        anyInt(),
                        any()
                    ),
                    times(1)
                );
            }
        }
    }
}
