/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_DECAY;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.MAX_RESULTS_RADIAL_RESCORING;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.Version;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.indices.ModelDao;

public class RNNQueryFactoryTests extends KNNTestCase {
    private static final String FILTER_FILED_NAME = "foo";
    private static final String FILTER_FILED_VALUE = "fooval";
    private static final QueryBuilder FILTER_QUERY_BUILDER = new TermQueryBuilder(FILTER_FILED_NAME, FILTER_FILED_VALUE);

    @Override
    public void setUp() throws Exception {
        super.setUp();
        RescoreRadialSearchQuery.initialize(new ExactSearcher(mock(ModelDao.OpenSearchKNNModelDao.class)));
    }

    private final int testQueryDimension = 17;
    private final float[] testQueryVector = new float[testQueryDimension];
    private final byte[] testByteQueryVector = new byte[testQueryDimension];
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final Float testRadius = 0.5f;
    private final int maxResultWindow = 20000;
    private final Map<String, ?> methodParameters = Map.of(METHOD_PARAMETER_EF_SEARCH, 100);

    // SQ 1-bit spec — the configuration that requires rescoring after radial search
    private ResolvedIndexSpec sqOneBitSpec() {
        return ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName("hnsw")
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(testQueryDimension)
            .indexVersionCreated(Version.CURRENT)
            .build();
    }

    // Non-quantized spec — isSQOneBit() == false, so no rescoring is required after radial search
    private ResolvedIndexSpec nonQuantizedSpec() {
        return ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName("hnsw")
            .encoderType(Encoder.EncoderType.FLAT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(testQueryDimension)
            .indexVersionCreated(Version.CURRENT)
            .build();
    }

    public void testCreate_whenLucene_withRadiusQuery_withFloatVector() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            Query query = RNNQueryFactory.create(
                knnEngine,
                testIndexName,
                testFieldName,
                testQueryVector,
                testRadius,
                DEFAULT_VECTOR_DATA_TYPE_FIELD
            );
            assertEquals(FloatVectorSimilarityQuery.class, query.getClass());
        }
    }

    public void testCreate_whenLucene_withRadiusQuery_withByteVector() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            MappedFieldType testMapper = mock(MappedFieldType.class);
            when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
            BitSetProducer parentFilter = mock(BitSetProducer.class);
            when(mockQueryShardContext.getParentFilter()).thenReturn(parentFilter);
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .byteVector(testByteQueryVector)
                .vectorDataType(VectorDataType.BYTE)
                .context(mockQueryShardContext)
                .filter(FILTER_QUERY_BUILDER)
                .build();
            Query query = RNNQueryFactory.create(createQueryRequest);
            assertEquals(ByteVectorSimilarityQuery.class, query.getClass());
        }
    }

    // Validates that the Lucene radial search path is actually taken and that the shared decay factor
    // (DEFAULT_LUCENE_RADIAL_SEARCH_DECAY = 0.95) is wired into the produced Lucene similarity query.
    // A mocked KNNVectorFieldType (non-quantized: isRescoringRequiredForRadial() == false) exercises the
    // real Lucene branch without the rescore wrapper. FloatVectorSimilarityQuery#equals compares the decay
    // field, so equality against a query constructed with the expected decay proves the value is wired.
    public void testCreate_whenLucene_thenDecayIsWiredIntoSimilarityQuery() {
        final KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        // Non-quantized spec: isSQOneBit() == false, so the rescore wrapper is not added and the
        // real Lucene similarity query is produced.
        when(mockFieldType.getResolvedSpec()).thenReturn(nonQuantizedSpec());

        final List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            // FLOAT -> FloatVectorSimilarityQuery
            final RNNQueryFactory.CreateQueryRequest floatRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .vectorDataType(VectorDataType.FLOAT)
                .vectorFieldType(mockFieldType)
                .build();
            final Query floatQuery = RNNQueryFactory.create(floatRequest);
            assertTrue("Lucene radial path must produce a FloatVectorSimilarityQuery", floatQuery instanceof FloatVectorSimilarityQuery);
            final FloatVectorSimilarityQuery expectedFloatQuery = new FloatVectorSimilarityQuery(
                testFieldName,
                testQueryVector,
                testRadius,
                DEFAULT_LUCENE_RADIAL_SEARCH_DECAY,
                null
            );
            assertEquals("Decay must be wired into the Lucene FloatVectorSimilarityQuery", expectedFloatQuery, floatQuery);

            // BYTE -> ByteVectorSimilarityQuery
            final RNNQueryFactory.CreateQueryRequest byteRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .byteVector(testByteQueryVector)
                .radius(testRadius)
                .vectorDataType(VectorDataType.BYTE)
                .vectorFieldType(mockFieldType)
                .build();
            final Query byteQuery = RNNQueryFactory.create(byteRequest);
            assertTrue("Lucene radial path must produce a ByteVectorSimilarityQuery", byteQuery instanceof ByteVectorSimilarityQuery);
            final ByteVectorSimilarityQuery expectedByteQuery = new ByteVectorSimilarityQuery(
                testFieldName,
                testByteQueryVector,
                testRadius,
                DEFAULT_LUCENE_RADIAL_SEARCH_DECAY,
                null
            );
            assertEquals("Decay must be wired into the Lucene ByteVectorSimilarityQuery", expectedByteQuery, byteQuery);
        }
    }

    public void testCreate_whenLucene_withFilter_thenSucceed() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            MappedFieldType testMapper = mock(MappedFieldType.class);
            when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .context(mockQueryShardContext)
                .filter(FILTER_QUERY_BUILDER)
                .radius(testRadius)
                .build();
            Query query = RNNQueryFactory.create(createQueryRequest);
            assertEquals(FloatVectorSimilarityQuery.class, query.getClass());
        }
    }

    public void testCreate_whenFaiss_thenSucceed() {
        // Given
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(mockQueryShardContext.getIndexSettings().getMaxResultWindow()).thenReturn(maxResultWindow);

        final KNNQuery expectedQuery = KNNQuery.builder()
            .field(testFieldName)
            .queryVector(testQueryVector)
            .indexName(testIndexName)
            .radius(testRadius)
            .methodParameters(methodParameters)
            .context(new KNNQuery.Context(maxResultWindow))
            .build();

        // When
        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .methodParameters(methodParameters)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        // Then
        assertEquals(expectedQuery, query);
    }

    // Verify that Faiss radial search with 32x SQ wraps the inner KNNQuery in RescoreRadialSearchQuery.
    public void testCreate_whenFaissSQ32x_thenWrapsInRescoreRadialSearchQuery() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(indexSettings.getMaxResultWindow()).thenReturn(maxResultWindow);
        when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
        assertTrue(rescoreQuery.getInnerQuery() instanceof KNNQuery);
        assertEquals(testFieldName, rescoreQuery.getField());
        assertEquals(testRadius, rescoreQuery.getRadius(), 0.0f);
        // maxResultsSize should come from IndexSettings.getMaxResultWindow()
    }

    // Given: memory-optimized Faiss radial search on a quantized field, with the coordinator-resolved
    // request window available
    // When: RNNQueryFactory creates the query
    // Then: the first pass is an oversampled top-k search bounded by the window, not an unbounded radial scan
    public void testCreate_whenMemoryOptimizedFaissSQ32x_thenBoundsFirstPassToRequestWindow() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(maxResultWindow);
        when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

        final RescoreContext rescoreContext = sqOneBitSpec().getRescoreContext();
        final int size = 25;
        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .memoryOptimizedSearchEnabled(true)
            .rescoreContext(rescoreContext)
            .size(size)
            .build();

        final Query query;
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any())).thenReturn(100);
            query = RNNQueryFactory.create(createQueryRequest);
        }

        assertTrue(query instanceof RescoreRadialSearchQuery);
        RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
        // 25 * 2 = 50, where 2x is the fixed oversample factor SQ 1-bit resolves to.
        final int expectedFirstPassK = (int) Math.ceil(size * rescoreContext.getOversampleFactor());
        assertEquals(50, expectedFirstPassK);
        assertEquals(expectedFirstPassK, rescoreQuery.getFirstPassK());
        // The first pass is now a bounded top-k query rather than a radial one. Memory-optimized Faiss
        // top-k search is wrapped in NativeEngineKnnVectorQuery.
        assertTrue(rescoreQuery.getInnerQuery() instanceof NativeEngineKnnVectorQuery);
        final KNNQuery firstPassQuery = ((NativeEngineKnnVectorQuery) rescoreQuery.getInnerQuery()).getKnnQuery();
        assertEquals(expectedFirstPassK, firstPassQuery.getK());
        assertNull(firstPassQuery.getRadius());
    }

    // Given: a request window large enough that window * oversample_factor exceeds MAX_FIRST_PASS_RESULTS
    // When: RNNQueryFactory creates the query
    // Then: the first pass is capped at MAX_FIRST_PASS_RESULTS * oversample_factor, so a deep page can still
    // surface MAX_FIRST_PASS_RESULTS documents after rescoring rather than silently returning fewer
    public void testCreate_whenRequestWindowExceedsFirstPassCap_thenCapScalesWithOversampleFactor() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(100_000);
        when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

        // The resolved SQ 1-bit rescore context, which is what KNNQueryBuilder#doToQuery passes in production.
        final RescoreContext rescoreContext = sqOneBitSpec().getRescoreContext();
        final int size = RescoreContext.MAX_FIRST_PASS_RESULTS * 2;
        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .memoryOptimizedSearchEnabled(true)
            .rescoreContext(rescoreContext)
            .size(size)
            .build();

        final Query query;
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            knnSettingsMockedStatic.when(() -> KNNSettings.getEfSearchParam(any())).thenReturn(100);
            query = RNNQueryFactory.create(createQueryRequest);
        }

        // SQ 1-bit pins the factor at 2x and disables the dimension-based override.
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.0f);
        final int expectedCap = (int) Math.ceil(
            RescoreContext.MAX_FIRST_PASS_RESULTS * RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR
        );
        final RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;

        assertEquals(expectedCap, rescoreQuery.getFirstPassK());
        assertTrue("the cap must scale past MAX_FIRST_PASS_RESULTS", expectedCap > RescoreContext.MAX_FIRST_PASS_RESULTS);
        // The window itself is not capped — rescoring truncates to it, and it is already bounded by max_result_window.
    }

    // Given: memory-optimized Faiss radial search on a quantized field, with no request window resolved
    // (an older coordinator, or a request the window processor declined to handle)
    // When: RNNQueryFactory creates the query
    // Then: it falls back to the unbounded radial first pass capped by max_result_window
    public void testCreate_whenMemoryOptimizedFaissSQ32xAndNoRequestWindow_thenFallsBackToMaxResultWindow() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(indexSettings.getMaxResultWindow()).thenReturn(maxResultWindow);
        when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .memoryOptimizedSearchEnabled(true)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
        assertEquals(maxResultWindow, rescoreQuery.getFirstPassK());
        assertEquals(testRadius.floatValue(), ((KNNQuery) rescoreQuery.getInnerQuery()).getRadius().floatValue(), 0.0f);
    }

    // Given: quantized Lucene radial search without a QueryShardContext and without a request window
    // When: RNNQueryFactory creates the query
    // Then: the radial rescore query is capped by the MAX_RESULTS_RADIAL_RESCORING fallback
    public void testCreate_whenRescoringRequiredAndNoContext_thenUsesRescoringFallbackLimit() {
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .vectorFieldType(mockFieldType)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        assertEquals(MAX_RESULTS_RADIAL_RESCORING, ((RescoreRadialSearchQuery) query).getFirstPassK());
    }

    // Given: quantized Lucene radial search with a custom max_result_window and no request window
    // When: RNNQueryFactory creates the query
    // Then: the radial rescore query is capped by the index's max_result_window
    public void testCreate_whenRescoringRequiredAndCustomMaxResultWindow_thenUsesMaxResultWindow() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(indexSettings.getMaxResultWindow()).thenReturn(500);
        when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .vectorFieldType(mockFieldType)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof RescoreRadialSearchQuery);
        assertEquals(500, ((RescoreRadialSearchQuery) query).getFirstPassK());
    }

    // Verify that quantized Lucene radial search wraps in the radial rescore query on every Lucene engine.
    public void testCreate_whenLuceneSQ32x_thenWrapsInRescoreRadialSearchQuery() {
        List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            KNNVectorFieldType mockFieldType = mock(KNNVectorFieldType.class);
            when(mockFieldType.getResolvedSpec()).thenReturn(sqOneBitSpec());

            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .vectorFieldType(mockFieldType)
                .build();

            Query query = RNNQueryFactory.create(createQueryRequest);

            assertTrue(query instanceof RescoreRadialSearchQuery);
        }
    }

    // Verify that non-quantized Faiss radial search returns bare KNNQuery (no wrapper).
    public void testCreate_whenFaissNotQuantized_thenNoWrapper() {
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        MappedFieldType testMapper = mock(MappedFieldType.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(mockQueryShardContext.getIndexSettings()).thenReturn(indexSettings);
        when(mockQueryShardContext.fieldMapper(any())).thenReturn(testMapper);
        when(indexSettings.getMaxResultWindow()).thenReturn(maxResultWindow);

        final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.FAISS)
            .indexName(testIndexName)
            .fieldName(testFieldName)
            .vector(testQueryVector)
            .radius(testRadius)
            .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
            .context(mockQueryShardContext)
            .build();

        Query query = RNNQueryFactory.create(createQueryRequest);

        assertTrue(query instanceof KNNQuery);
        assertFalse(query instanceof RescoreRadialSearchQuery);
    }

    // Verify that non-quantized Lucene radial search returns bare FloatVectorSimilarityQuery (no wrapper).
    public void testCreate_whenLuceneNotQuantized_thenNoWrapper() {
        List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                .vectorDataType(DEFAULT_VECTOR_DATA_TYPE_FIELD)
                .build();

            Query query = RNNQueryFactory.create(createQueryRequest);

            assertTrue(query instanceof FloatVectorSimilarityQuery);
            assertFalse(query instanceof RescoreRadialSearchQuery);
        }
    }

    // Verify that createLuceneRadialQuery throws IllegalArgumentException for unsupported vector data types.
    // The default branch in the switch statement (line 146-152) should be hit when a BINARY vector type
    // is passed to the Lucene radial search path, since only FLOAT and BYTE are supported.
    public void testCreate_whenLuceneWithUnsupportedVectorDataType_thenThrows() {
        List<KNNEngine> luceneEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());

        for (KNNEngine knnEngine : luceneEngines) {
            final RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(testIndexName)
                .fieldName(testFieldName)
                .vector(testQueryVector)
                .radius(testRadius)
                // BINARY is not supported for Lucene radial search — should hit the default branch
                .vectorDataType(VectorDataType.BINARY)
                .build();

            expectThrows(IllegalArgumentException.class, () -> RNNQueryFactory.create(createQueryRequest));
        }
    }
}
