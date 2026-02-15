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
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.utils.TopDocsTestUtils.*;

public class ExplainTests extends KNNWeightTestCase {

    @Mock
    private Weight filterQueryWeight;
    @Mock
    private LeafReaderContext leafReaderContext;

    private void setupTest(final int[] filterDocIds, final Map<String, String> attributesMap) throws IOException {
        setupTest(filterDocIds, attributesMap, filterDocIds != null ? filterDocIds.length : 0, SpaceType.L2, true, null, null, null);
    }

    private void setupTest(
        final int[] filterDocIds,
        final Map<String, String> attributesMap,
        final int maxDoc,
        final SpaceType spaceType,
        final boolean isCompoundFile,
        final byte[] byteVector,
        final float[] floatVector,
        final MockedStatic<KNNVectorValuesFactory> vectorValuesFactoryMockedStatic
    ) throws IOException {

        final Scorer filterScorer = mock(Scorer.class);
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);

        Bits liveDocsBits = null;
        if (filterDocIds != null) {
            FixedBitSet filterBitSet = new FixedBitSet(filterDocIds.length);
            for (int docId : filterDocIds) {
                filterBitSet.set(docId);
            }
            liveDocsBits = mock(Bits.class);
            for (int filterDocId : filterDocIds) {
                when(liveDocsBits.get(filterDocId)).thenReturn(true);
            }
            when(liveDocsBits.length()).thenReturn(1000);

            when(filterQueryWeight.scorer(leafReaderContext)).thenReturn(filterScorer);
            when(filterScorer.iterator()).thenReturn(DocIdSetIterator.all(filterDocIds.length + 1));
        }
        final SegmentReader reader = mockSegmentReader(isCompoundFile);
        when(reader.maxDoc()).thenReturn(maxDoc);
        when(reader.getLiveDocs()).thenReturn(liveDocsBits);

        when(leafReaderContext.reader()).thenReturn(reader);
        when(leafReaderContext.id()).thenReturn(new Object());

        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
        when(fieldInfo.attributes()).thenReturn(attributesMap);
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());
        when(fieldInfo.getAttribute(VECTOR_DATA_TYPE_FIELD)).thenReturn(
            byteVector != null ? VectorDataType.BINARY.getValue() : VectorDataType.FLOAT.getValue()
        );
        when(fieldInfo.getName()).thenReturn(FIELD_NAME);

        if (floatVector != null) {
            final BinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(List.of(floatVector));
            when(reader.getBinaryDocValues(FIELD_NAME)).thenReturn(binaryDocValues);
        }

        if (byteVector != null) {
            final KNNVectorValues vectorValues = TestVectorValues.createKNNBinaryVectorValues(
                new TestVectorValues.PredefinedByteVectorBinaryDocValues(List.of(byteVector))
            );
            vectorValuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(vectorValues);
        }
    }

    private void assertExplanation(Explanation explanation, float expectedScore, String topSearch, String... leafDescription) {
        assertNotNull(explanation);
        assertTrue(explanation.isMatch());
        assertEquals(expectedScore, explanation.getValue().floatValue(), 0.01f);
        assertTrue(explanation.getDescription().contains(topSearch));
        assertEquals(1, explanation.getDetails().length);
        Explanation explanationDetail = explanation.getDetails()[0];
        assertEquals(expectedScore, explanation.getValue().floatValue(), 0.01f);
        for (String description : leafDescription) {
            assertTrue(explanationDetail.getDescription().contains(description));
        }
    }

    private void assertDiskSearchExplanation(Explanation explanation, String[] topSearchDesc, String... leafDescription) {
        assertNotNull(explanation);
        assertTrue(explanation.isMatch());
        for (String description : topSearchDesc) {
            assertTrue(explanation.getDescription().contains(description));
        }
        assertEquals(1, explanation.getDetails().length);
        Explanation explanationDetail = explanation.getDetails()[0];
        for (String description : leafDescription) {
            assertTrue(explanationDetail.getDescription().contains(description));
        }
    }

    @SneakyThrows
    public void testDiskBasedSearchWithShardRescoringEnabledANN() {
        int k = 3;
        knnSettingsMockedStatic.when(() -> KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(INDEX_NAME)).thenReturn(false);

        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), eq(null), anyInt(), any())
        ).thenReturn(getFilteredKNNQueryResults());

        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(RescoreContext.MIN_OVERSAMPLE_FACTOR - 1).build();

        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };

        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        setupTest(filterDocIds, attributesMap);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(k)
            .indexName(INDEX_NAME)
            .filterQuery(FILTER_QUERY)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .rescoreContext(rescoreContext)
            .explain(true)
            .build();
        query.setExplain(true);

        final float boost = 1;
        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);

        // When
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Then
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(FILTERED_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        jniServiceMockedStatic.verify(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
            times(1)
        );

        final List<Integer> actualDocIds = new ArrayList<>();
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = translatedScores.get(docId) * boost;
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            String[] expectedTopDescription = new String[] {
                KNNConstants.DISK_BASED_SEARCH,
                "the first pass k was " + rescoreContext.getFirstPassK(k, false, QUERY_VECTOR.length),
                "over sampling factor of " + rescoreContext.getOversampleFactor(),
                "with vector dimension of " + QUERY_VECTOR.length,
                "shard level rescoring enabled" };
            assertDiskSearchExplanation(
                explanation,
                expectedTopDescription,
                ANN_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue()
            );
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testDiskBasedSearchWithShardRescoringDisabledExact() {
        knnSettingsMockedStatic.when(() -> KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(INDEX_NAME)).thenReturn(true);
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(RescoreContext.MAX_OVERSAMPLE_FACTOR - 1).build();

        ExactSearcher mockedExactSearcher = mock(ExactSearcher.class);
        KNNWeight.initialize(null, mockedExactSearcher);

        final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
        final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);

        Map<String, String> attributesMap = Map.of(
            SPACE_TYPE,
            spaceType.getValue(),
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        setupTest(null, attributesMap, 1, spaceType, false, null, null, null);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(queryVector)
            .indexName(INDEX_NAME)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .rescoreContext(rescoreContext)
            .explain(true)
            .build();
        final KNNWeight knnWeight = new DefaultKNNWeight(query, 1.0f, null);

        final ExactSearcher.ExactSearcherContext exactSearchContext = ExactSearcher.ExactSearcherContext.builder()
            // setting to true, so that if quantization details are present we want to do search on the quantized
            // vectors as this flow is used in first pass of search.
            .useQuantizedVectorsForSearch(true)
            .floatQueryVector(queryVector)
            .field(FIELD_NAME)
            .isMemoryOptimizedSearchEnabled(false)
            .build();
        when(mockedExactSearcher.searchLeaf(leafReaderContext, exactSearchContext)).thenReturn(buildTopDocs(DOC_ID_TO_SCORES));

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);

        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = DOC_ID_TO_SCORES.get(docId);
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            String[] expectedTopDescription = new String[] {
                KNNConstants.DISK_BASED_SEARCH,
                "the first pass k was " + rescoreContext.getFirstPassK(0, true, queryVector.length),
                "over sampling factor of " + rescoreContext.getOversampleFactor(),
                "with vector dimension of " + queryVector.length,
                "shard level rescoring disabled" };
            assertDiskSearchExplanation(
                explanation,
                expectedTopDescription,
                EXACT_SEARCH,
                VectorDataType.FLOAT.name(),
                spaceType.getValue(),
                "no native engine files"
            );
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
        // verify JNI Service is not called
        jniServiceMockedStatic.verifyNoInteractions();
        verify(mockedExactSearcher).searchLeaf(leafReaderContext, exactSearchContext);
    }

    @SneakyThrows
    public void testDefaultANNSearch() {
        // Given
        int k = 3;
        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), eq(null), anyInt(), any())
        ).thenReturn(getFilteredKNNQueryResults());

        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        setupTest(filterDocIds, attributesMap);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(k)
            .indexName(INDEX_NAME)
            .filterQuery(FILTER_QUERY)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .explain(true)
            .build();
        query.setExplain(true);

        final float boost = 1;

        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);

        // When
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Then
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(FILTERED_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        jniServiceMockedStatic.verify(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
            times(1)
        );

        final List<Integer> actualDocIds = new ArrayList<>();
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = translatedScores.get(docId) * boost;
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                ANN_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                SpaceType.L2.explainScoreTranslation(DOC_ID_TO_SCORES.get(docId))
            );
            Explanation nestedDetail = explanation.getDetails()[0].getDetails()[0];
            assertTrue(nestedDetail.getDescription().contains(KNNEngine.FAISS.name()));
            assertEquals(DOC_ID_TO_SCORES.get(docId), nestedDetail.getValue().floatValue(), 0.01f);
            assertEquals(score, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testANN_FilteredExactSearchAfterANN() {
        ExactSearcher mockedExactSearcher = mock(ExactSearcher.class);
        KNNWeight.initialize(null, mockedExactSearcher);
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        when(mockedExactSearcher.searchLeaf(any(), any())).thenReturn(buildTopDocs(translatedScores));
        // Given
        int k = 4;
        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), eq(null), anyInt(), any())
        ).thenReturn(getFilteredKNNQueryResults());

        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        setupTest(filterDocIds, attributesMap);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(k)
            .indexName(INDEX_NAME)
            .filterQuery(FILTER_QUERY)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .explain(true)
            .build();
        query.setExplain(true);

        final float boost = 1;
        KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);

        // When
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Then
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        jniServiceMockedStatic.verify(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
            times(1)
        );

        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = translatedScores.get(docId) * boost;
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                EXACT_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                "since the number of documents returned are less than K",
                "there are more than K filtered Ids"
            );
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testANN_whenNoEngineFiles_thenPerformExactSearch() {
        ExactSearcher mockedExactSearcher = mock(ExactSearcher.class);
        final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
        final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);
        KNNWeight.initialize(null, mockedExactSearcher);
        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(queryVector)
            .indexName(INDEX_NAME)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .explain(true)
            .build();
        final KNNWeight knnWeight = new DefaultKNNWeight(query, 1.0f, null);

        Map<String, String> attributesMap = Map.of(
            SPACE_TYPE,
            spaceType.getValue(),
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        setupTest(null, attributesMap, 1, spaceType, false, null, null, null);

        final ExactSearcher.ExactSearcherContext exactSearchContext = ExactSearcher.ExactSearcherContext.builder()
            // setting to true, so that if quantization details are present we want to do search on the quantized
            // vectors as this flow is used in first pass of search.
            .useQuantizedVectorsForSearch(true)
            .field(FIELD_NAME)
            .floatQueryVector(queryVector)
            .isMemoryOptimizedSearchEnabled(false)
            .build();
        when(mockedExactSearcher.searchLeaf(leafReaderContext, exactSearchContext)).thenReturn(buildTopDocs(DOC_ID_TO_SCORES));
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = DOC_ID_TO_SCORES.get(docId);
            assertEquals(score, knnScorer.score(), 0.00000001f);
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                EXACT_SEARCH,
                VectorDataType.FLOAT.name(),
                spaceType.getValue(),
                "no native engine files"
            );
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
        // verify JNI Service is not called
        jniServiceMockedStatic.verifyNoInteractions();
        verify(mockedExactSearcher).searchLeaf(leafReaderContext, exactSearchContext);
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenFTVGreaterThanFilterId() {

        KNNWeight.initialize(null);
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(10);
        byte[] vector = new byte[] { 1, 3 };
        int k = 1;
        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.HAMMING.name(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "BHNSW32")
        );

        try (MockedStatic<KNNVectorValuesFactory> vectorValuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            setupTest(filterDocIds, attributesMap, 100, SpaceType.HAMMING, true, vector, null, vectorValuesFactoryMockedStatic);
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

            query.setExplain(true);
            final float boost = (float) randomDoubleBetween(0, 10, true);
            final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);

            final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
            assertNotNull(knnScorer);
            knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
            final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
            assertNotNull(docIdSetIterator);
            assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

            final List<Integer> actualDocIds = new ArrayList<>();
            for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
                actualDocIds.add(docId);
                float score = BINARY_EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost;
                assertEquals(score, knnScorer.score(), 0.01f);
                Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
                assertExplanation(
                    explanation,
                    score,
                    ANN_SEARCH,
                    EXACT_SEARCH,
                    VectorDataType.BINARY.name(),
                    SpaceType.HAMMING.getValue(),
                    "is greater than or equal to estimated distance computations",
                    "since filtered threshold value"
                );
            }
            assertEquals(docIdSetIterator.cost(), actualDocIds.size());
            assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
        }
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenMDCGreaterThanFilterId() {
        ModelDao modelDao = mock(ModelDao.class);
        KNNWeight.initialize(modelDao);
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(-1);
        float[] vector = new float[] { 0.1f, 0.3f };
        int k = 1;
        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.name(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        setupTest(filterDocIds, attributesMap, 100, SpaceType.L2, true, null, vector, null);

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, k, INDEX_NAME, FILTER_QUERY, null, null);
        query.setExplain(true);

        final float boost = (float) randomDoubleBetween(0, 10, true);
        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost;
            assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                EXACT_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                "since max distance computation",
                "is greater than or equal to estimated distance computations"
            );
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testANNWithFilterQuery_whenFilterIdLessThanK() {
        ModelDao modelDao = mock(ModelDao.class);
        KNNWeight.initialize(modelDao);
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(-1);
        float[] vector = new float[] { 0.1f, 0.3f };
        final int[] filterDocIds = new int[] { 0 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.name(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        setupTest(filterDocIds, attributesMap, 100, SpaceType.L2, true, null, vector, null);

        final KNNQuery query = new KNNQuery(FIELD_NAME, QUERY_VECTOR, K, INDEX_NAME, FILTER_QUERY, null, null);
        query.setExplain(true);

        final float boost = 1;
        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost;
            assertEquals(EXACT_SEARCH_DOC_ID_TO_SCORES.get(docId) * boost, knnScorer.score(), 0.01f);
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                EXACT_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                "since filteredIds",
                "is less than or equal to K"
            );
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testFilteredANNSearch_exactSearchDisabled_thenPerformANNSearch() {
        // Given
        int k = 4;
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(1);
        knnSettingsMockedStatic.when(() -> KNNSettings.isKnnIndexFaissEfficientFilterExactSearchDisabled(INDEX_NAME)).thenReturn(true);
        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), eq(null), anyInt(), any())
        ).thenReturn(getFilteredKNNQueryResults());

        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        setupTest(filterDocIds, attributesMap);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(k)
            .indexName(INDEX_NAME)
            .filterQuery(FILTER_QUERY)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .explain(true)
            .build();
        query.setExplain(true);

        final float boost = 1;

        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);

        // When
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        // Then
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(FILTERED_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        jniServiceMockedStatic.verify(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
            times(1)
        );

        final List<Integer> actualDocIds = new ArrayList<>();
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = translatedScores.get(docId) * boost;
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);

            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                ANN_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                SpaceType.L2.explainScoreTranslation(DOC_ID_TO_SCORES.get(docId)),
                ", it is not falling back to exact search after ",
                ANN_SEARCH,
                " search since exact search is disabled,"
            );
            Explanation nestedDetail = explanation.getDetails()[0].getDetails()[0];
            assertTrue(nestedDetail.getDescription().contains(KNNEngine.FAISS.name()));
            assertEquals(DOC_ID_TO_SCORES.get(docId), nestedDetail.getValue().floatValue(), 0.01f);
            assertEquals(score, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testFilteredANNSearch_cardinalityAboveThreshold_thenPerformANNSearch() {
        int k = 4;
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(1);
        knnSettingsMockedStatic.when(() -> KNNSettings.getKnnIndexFaissEfficientFilterDisableExactSearchThreshold(INDEX_NAME))
            .thenReturn(3);
        jniServiceMockedStatic.when(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), eq(null), anyInt(), any())
        ).thenReturn(getFilteredKNNQueryResults());

        final int[] filterDocIds = new int[] { 0, 1, 2, 3, 4, 5 };
        final Map<String, String> attributesMap = ImmutableMap.of(
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        setupTest(filterDocIds, attributesMap);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(QUERY_VECTOR)
            .k(k)
            .indexName(INDEX_NAME)
            .filterQuery(FILTER_QUERY)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .vectorDataType(VectorDataType.FLOAT)
            .explain(true)
            .build();
        query.setExplain(true);

        final float boost = 1;

        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, filterQueryWeight);
        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);

        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        assertNotNull(docIdSetIterator);
        assertEquals(FILTERED_DOC_ID_TO_SCORES.size(), docIdSetIterator.cost());

        jniServiceMockedStatic.verify(
            () -> JNIService.queryIndex(anyLong(), eq(QUERY_VECTOR), eq(k), eq(HNSW_METHOD_PARAMETERS), any(), any(), anyInt(), any()),
            times(1)
        );

        final List<Integer> actualDocIds = new ArrayList<>();
        final Map<Integer, Float> translatedScores = getTranslatedScores(SpaceType.L2::scoreTranslation);
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = translatedScores.get(docId) * boost;
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);

            assertExplanation(
                explanation,
                score,
                ANN_SEARCH,
                ANN_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                SpaceType.L2.explainScoreTranslation(DOC_ID_TO_SCORES.get(docId)),
                ", it is not falling back to exact search after ",
                ANN_SEARCH,
                " search since filter cardinality = "
            );
            Explanation nestedDetail = explanation.getDetails()[0].getDetails()[0];
            assertTrue(nestedDetail.getDescription().contains(KNNEngine.FAISS.name()));
            assertEquals(DOC_ID_TO_SCORES.get(docId), nestedDetail.getValue().floatValue(), 0.01f);
            assertEquals(score, knnScorer.score(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testRadialANNSearch() {
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

        Map<String, String> attributesMap = Map.of(
            SPACE_TYPE,
            SpaceType.L2.getValue(),
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        setupTest(null, attributesMap);

        KNNQuery.Context context = mock(KNNQuery.Context.class);
        when(context.getMaxResultWindow()).thenReturn(maxResults);
        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(queryVector)
            .radius(radius)
            .indexName(INDEX_NAME)
            .context(context)
            .explain(true)
            .vectorDataType(VectorDataType.FLOAT)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .build();
        final float boost = 1;
        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, null);

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
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
            float score = translatedScores.get(docId) * boost;
            assertEquals(score, knnScorer.score(), 0.01f);
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(
                explanation,
                score,
                RADIAL_SEARCH,
                ANN_SEARCH,
                VectorDataType.FLOAT.name(),
                SpaceType.L2.getValue(),
                SpaceType.L2.explainScoreTranslation(DOC_ID_TO_SCORES.get(docId))
            );
            Explanation nestedDetail = explanation.getDetails()[0].getDetails()[0];
            assertTrue(nestedDetail.getDescription().contains(KNNEngine.FAISS.name()));
            assertEquals(DOC_ID_TO_SCORES.get(docId), nestedDetail.getValue().floatValue(), 0.01f);
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
    }

    @SneakyThrows
    public void testRadialExactSearch() {
        ExactSearcher mockedExactSearcher = mock(ExactSearcher.class);
        final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);
        KNNWeight.initialize(null, mockedExactSearcher);

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

        Map<String, String> attributesMap = Map.of(
            SPACE_TYPE,
            spaceType.getValue(),
            KNN_ENGINE,
            KNNEngine.FAISS.getName(),
            PARAMETERS,
            String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
        );

        setupTest(null, attributesMap, 0, spaceType, false, null, null, null);

        KNNQuery.Context context = mock(KNNQuery.Context.class);
        when(context.getMaxResultWindow()).thenReturn(maxResults);

        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(queryVector)
            .radius(radius)
            .indexName(INDEX_NAME)
            .context(context)
            .explain(true)
            .vectorDataType(VectorDataType.FLOAT)
            .methodParameters(HNSW_METHOD_PARAMETERS)
            .build();
        final float boost = 1;
        final KNNWeight knnWeight = new DefaultKNNWeight(query, boost, null);
        final ExactSearcher.ExactSearcherContext exactSearchContext = ExactSearcher.ExactSearcherContext.builder()
            // setting to true, so that if quantization details are present we want to do search on the quantized
            // vectors as this flow is used in first pass of search.
            .useQuantizedVectorsForSearch(true)
            .floatQueryVector(queryVector)
            .field(FIELD_NAME)
            .radius(radius)
            .isMemoryOptimizedSearchEnabled(false)
            .maxResultWindow(maxResults)
            .build();
        when(mockedExactSearcher.searchLeaf(leafReaderContext, exactSearchContext)).thenReturn(buildTopDocs(DOC_ID_TO_SCORES));

        final KNNScorer knnScorer = (KNNScorer) knnWeight.scorer(leafReaderContext);
        assertNotNull(knnScorer);
        knnWeight.getKnnExplanation().addKnnScorer(leafReaderContext, knnScorer);
        final DocIdSetIterator docIdSetIterator = knnScorer.iterator();
        final List<Integer> actualDocIds = new ArrayList<>();
        for (int docId = docIdSetIterator.nextDoc(); docId != NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
            actualDocIds.add(docId);
            float score = DOC_ID_TO_SCORES.get(docId) * boost;
            assertEquals(score, knnScorer.score(), 0.01f);
            Explanation explanation = knnWeight.explain(leafReaderContext, docId, score);
            assertExplanation(explanation, score, RADIAL_SEARCH, EXACT_SEARCH, VectorDataType.FLOAT.name(), spaceType.getValue());
        }
        assertEquals(docIdSetIterator.cost(), actualDocIds.size());
        assertTrue(Comparators.isInOrder(actualDocIds, Comparator.naturalOrder()));
        // verify JNI Service is not called
        jniServiceMockedStatic.verifyNoInteractions();
        verify(mockedExactSearcher).searchLeaf(leafReaderContext, exactSearchContext);
    }
}
