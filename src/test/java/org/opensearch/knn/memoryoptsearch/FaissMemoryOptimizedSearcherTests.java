/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.generate.IndexingType;
import org.opensearch.knn.generate.SearchTestHelper;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.nativeindex.MemoryOptimizedSearchIndexingSupport;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.query.FilterIdsSelector;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.query.MemoryOptimizedSearchScoreConverter;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.index.vectorvalues.VectorValueExtractorStrategy;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Constructor;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QTY;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.generate.SearchTestHelper.convertToFloatArray;
import static org.opensearch.knn.generate.SearchTestHelper.generateOneSingleByteVector;
import static org.opensearch.knn.generate.SearchTestHelper.generateOneSingleFloatVector;
import static org.opensearch.knn.generate.SearchTestHelper.generateRandomByteVectors;
import static org.opensearch.knn.generate.SearchTestHelper.getKnnAnswerSetForVectors;

public class FaissMemoryOptimizedSearcherTests extends KNNTestCase {
    private static final String TARGET_FIELD = "target_field";
    private static final String FLOAT_HNSW_INDEX_DESCRIPTION = "HNSW16,Flat";
    private static final Map<String, Object> FLOAT32_ENCODER_PARAMETERS = Map.of(NAME, ENCODER_FLAT);
    private static final String BYTE_HNSW_INDEX_DESCRIPTION = "HNSW16,SQ8_direct_signed";
    private static final String FLOAT16_HNSW_INDEX_DESCRIPTION = "HNSW16,SQfp16";
    private static final Map<String, Object> FLOAT16_ENCODER_PARAMETERS = Map.of(
        NAME,
        ENCODER_SQ,
        PARAMETERS,
        Map.of(TYPE, FAISS_SQ_ENCODER_FP16, FAISS_SQ_CLIP, false)
    );
    private static final String BINARY_HSNW_INDEX_DESCRIPTION = "BHNSW16,Flat";
    private static final Map<String, Object> EMTPY_ENCODER_PARAMETERS = Map.of();
    private static final int DIMENSIONS = 128;
    private static final int TOTAL_NUM_DOCS_IN_SEGMENT = 300;
    private static final int TOP_K = 30;
    private static final float NO_FILTERING = Float.NaN;
    private static final FlatVectorsScorer SCORER = new NativeEngines990KnnVectorsScorer(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );

    public void test32xQuantizedBinaryIndexType() {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.BINARY,
            BINARY_HSNW_INDEX_DESCRIPTION,
            -1000000,
            1000000,
            FLOAT32_ENCODER_PARAMETERS
        );
        testingSpec.quantizationParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        testingSpec.quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    public void test16xQuantizedBinaryIndexType() {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.BINARY,
            BINARY_HSNW_INDEX_DESCRIPTION,
            -1000000,
            1000000,
            FLOAT32_ENCODER_PARAMETERS
        );
        testingSpec.quantizationParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        testingSpec.quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build();

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    public void test8xQuantizedBinaryIndexType() {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.BINARY,
            BINARY_HSNW_INDEX_DESCRIPTION,
            -1000000,
            1000000,
            FLOAT32_ENCODER_PARAMETERS
        );
        testingSpec.quantizationParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.FOUR_BIT).build();
        testingSpec.quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build();

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    public void testFloatIndexType() {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.FLOAT,
            FLOAT_HNSW_INDEX_DESCRIPTION,
            -10,
            10,
            FLOAT32_ENCODER_PARAMETERS
        );

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    public void testByteIndexType() {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.BYTE,
            BYTE_HNSW_INDEX_DESCRIPTION,
            -128,
            127,
            EMTPY_ENCODER_PARAMETERS
        );

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    public void testFloat16IndexType() {
        // Validate mmap optimized logic for FP16.
        doTestFloat16IndexType(false);
    }

    public void testFloat16IndexTypeWithNIOFSDirectory() {
        // For FP16, it applies a different logic for non-mmap Directory.
        // Therefore, configuring NIOFSDirectory to validate the logic.
        doTestFloat16IndexType(true);
    }

    public void doTestFloat16IndexType(final boolean useNIOFSDirectory) {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.FLOAT,
            FLOAT16_HNSW_INDEX_DESCRIPTION,
            -10,
            10,
            FLOAT16_ENCODER_PARAMETERS
        );

        if (useNIOFSDirectory) {
            testingSpec.directoryClass = NIOFSDirectory.class;
        }

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    private void doTestADCWithBinaryQuantization(final SpaceType spaceType) {
        final TestingSpec adcEnabledSpec = new TestingSpec(
            VectorDataType.BINARY,
            BINARY_HSNW_INDEX_DESCRIPTION,
            -100,
            100,
            FLOAT32_ENCODER_PARAMETERS
        );
        adcEnabledSpec.quantizationParams = ScalarQuantizationParams.builder()
            .sqType(ScalarQuantizationType.ONE_BIT)
            .enableADC(true)
            .build();
        adcEnabledSpec.quantizationConfig = QuantizationConfig.builder()
            .quantizationType(ScalarQuantizationType.ONE_BIT)
            .enableADC(true)
            .build();

        doSearchTest(adcEnabledSpec, IndexingType.DENSE, spaceType, true, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.DENSE, spaceType, true, NO_FILTERING);
        doSearchTest(adcEnabledSpec, IndexingType.DENSE, spaceType, false, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.DENSE, spaceType, false, NO_FILTERING);

        doSearchTest(adcEnabledSpec, IndexingType.DENSE_NESTED, spaceType, true, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.DENSE_NESTED, spaceType, true, NO_FILTERING);
        doSearchTest(adcEnabledSpec, IndexingType.DENSE_NESTED, spaceType, false, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.DENSE_NESTED, spaceType, false, NO_FILTERING);

        doSearchTest(adcEnabledSpec, IndexingType.SPARSE, spaceType, true, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.SPARSE, spaceType, true, NO_FILTERING);
        doSearchTest(adcEnabledSpec, IndexingType.SPARSE, spaceType, false, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.SPARSE, spaceType, false, NO_FILTERING);

        doSearchTest(adcEnabledSpec, IndexingType.SPARSE_NESTED, spaceType, true, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.SPARSE_NESTED, spaceType, true, NO_FILTERING);
        doSearchTest(adcEnabledSpec, IndexingType.SPARSE_NESTED, spaceType, false, 0.8f);
        doSearchTest(adcEnabledSpec, IndexingType.SPARSE_NESTED, spaceType, false, NO_FILTERING);
    }

    public void testADCWithBinaryQuantizationL2() {
        doTestADCWithBinaryQuantization(SpaceType.L2);
    }

    public void testADCWithBinaryQuantizationIP() {
        doTestADCWithBinaryQuantization(SpaceType.INNER_PRODUCT);
    }

    public void testADCWithBinaryQuantizationCosine() {
        doTestADCWithBinaryQuantization(SpaceType.COSINESIMIL);
    }

    @SneakyThrows
    public void testGetByteVectorValues_returnsDifferentInstancesPerCall() {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        FaissIdMapIndex faissIndex = mock(FaissIdMapIndex.class);
        Mockito.when(faissIndex.getByteValues(any())).thenReturn(mock(ByteVectorValues.class));
        Mockito.when(faissIndex.getVectorSimilarityFunction()).thenReturn(SpaceType.L2.getKnnVectorSimilarityFunction());
        Mockito.when(faissIndex.getFaissHnsw()).thenReturn(mock(FaissHNSW.class));
        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(
            mock(IndexInput.class),
            faissIndex,
            fieldInfo,
            FlatVectorsScorerProvider.getFlatVectorsScorer(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, SCORER)
        );

        final var first = searcher.getByteVectorValues(mock(KnnVectorValues.DocIndexIterator.class));
        final var second = searcher.getByteVectorValues(mock(KnnVectorValues.DocIndexIterator.class));
        assertNotSame("getByteVectorValues() should return a new instance per call", first, second);
    }

    @SneakyThrows
    public void testWarmUp_byteVectorEncoding_warmsUpByteValues() {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

        FaissIdMapIndex faissIndex = mock(FaissIdMapIndex.class);
        Mockito.when(faissIndex.getVectorSimilarityFunction()).thenReturn(SpaceType.L2.getKnnVectorSimilarityFunction());
        Mockito.when(faissIndex.getFaissHnsw()).thenReturn(mock(FaissHNSW.class));
        Mockito.when(faissIndex.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);

        ByteVectorValues mockByteValues = mock(ByteVectorValues.class);
        when(mockByteValues.size()).thenReturn(3);
        Mockito.when(faissIndex.getByteValues(any())).thenReturn(mockByteValues);

        IndexInput mockIndexInput = mock(IndexInput.class);
        when(mockIndexInput.clone()).thenReturn(mockIndexInput);
        when(mockIndexInput.length()).thenReturn(0L);

        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(
            mockIndexInput,
            faissIndex,
            fieldInfo,
            FlatVectorsScorerProvider.getFlatVectorsScorer(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, SCORER)
        );

        searcher.warmUp();

        // Verify byte vector values were warmed up (readAll iterates through all values)
        for (int i = 0; i < 3; i++) {
            Mockito.verify(mockByteValues).vectorValue(i);
        }
    }

    @SneakyThrows
    private void doSearchTest(final TestingSpec testingSpec, final IndexingType indexingType) {
        final List<SpaceType> spaceTypes;
        if (testingSpec.dataType == VectorDataType.BINARY) {
            spaceTypes = Arrays.asList(SpaceType.HAMMING);
        } else if (testingSpec.dataType == VectorDataType.BYTE) {
            // Byte vectors cannot be L2-normalized, so cosine similarity is not supported.
            spaceTypes = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT);
        } else {
            spaceTypes = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);
        }

        for (final SpaceType spaceType : spaceTypes) {
            doSearchTest(testingSpec, indexingType, spaceType, false, NO_FILTERING);
            doSearchTest(testingSpec, indexingType, spaceType, false, 0.8f);

            doSearchTest(testingSpec, indexingType, spaceType, false, NO_FILTERING);
            doSearchTest(testingSpec, indexingType, spaceType, false, 0.8f);

            doSearchTest(testingSpec, indexingType, spaceType, true, NO_FILTERING);
            doSearchTest(testingSpec, indexingType, spaceType, true, 0.8f);

            doSearchTest(testingSpec, indexingType, spaceType, true, NO_FILTERING);
            doSearchTest(testingSpec, indexingType, spaceType, true, 0.8f);
        }
    }

    @SneakyThrows
    private void doSearchTest(
        final TestingSpec testingSpec,
        final IndexingType indexingType,
        final SpaceType spaceType,
        final boolean doExhaustiveSearch,
        final float filteringRatio
    ) {
        // Build FAISS index
        final BuildInfo buildInfo = buildFaissIndex(testingSpec, TOTAL_NUM_DOCS_IN_SEGMENT, indexingType, spaceType);

        // Load FAISS index via JNI
        long indexPointer = -1;
        try (final Directory directory = newFSDirectory(buildInfo.tempDirPath)) {
            try (final IndexInput input = directory.openInput(buildInfo.faissIndexFile, IOContext.READONCE)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(input);
                if (testingSpec.quantizationConfig != null && testingSpec.quantizationConfig.enableADC) {
                    buildInfo.parameters.put("data_type", VectorDataType.FLOAT.getValue());
                    buildInfo.parameters.put(ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER, true);
                    buildInfo.parameters.put("quantization_level", "ScalarQuantizationParams_1");
                    buildInfo.parameters.put("space_type", spaceType.getValue());
                    indexPointer = JNIService.loadIndex(indexInputWithBuffer, buildInfo.parameters, KNNEngine.FAISS);
                } else {
                    indexPointer = JNIService.loadIndex(indexInputWithBuffer, buildInfo.parameters, KNNEngine.FAISS);
                }
            }
        }

        assertNotEquals(-1, indexPointer);

        // Make filtered ids
        long[] filteredIds = null;
        if (Float.compare(filteringRatio, NO_FILTERING) != 0) {
            // Take only X%. Ex: Keep 80% of docs = cut off 20% of docs.
            final List<Integer> filteredDocIds = SearchTestHelper.takePortions(buildInfo.documentIds, filteringRatio);
            filteredIds = filteredDocIds.stream().mapToLong(Integer::longValue).toArray();
        }

        // Reconstruct parent ids if it's necessary
        int[] parentIds = null;
        if (indexingType.isNested()) {
            parentIds = SearchTestHelper.extractParentIds(buildInfo.documentIds);
        }

        // Take top-k results
        final int k = TOP_K;

        // Start search via JNI
        Object queryForVectorReader = null;
        Object query = null;
        byte[] byteQuery;
        final KNNQueryResult[] resultsFromFaiss;

        if (testingSpec.dataType == VectorDataType.FLOAT || testingSpec.dataType == VectorDataType.BYTE) {
            if (testingSpec.dataType == VectorDataType.FLOAT) {
                queryForVectorReader = query = generateOneSingleFloatVector(
                    DIMENSIONS,
                    testingSpec.minValue,
                    testingSpec.maxValue,
                    spaceType == SpaceType.COSINESIMIL
                );
            } else if (testingSpec.dataType == VectorDataType.BYTE) {
                queryForVectorReader = byteQuery = generateOneSingleByteVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);
                query = convertToFloatArray(byteQuery);
            }

            resultsFromFaiss = JNIService.queryIndex(
                indexPointer,
                (float[]) query,
                k,
                buildInfo.parameters,
                KNNEngine.FAISS,
                filteredIds,
                FilterIdsSelector.FilterIdsSelectorType.BATCH.getValue(),
                parentIds
            );
        } else if (testingSpec.quantizationConfig != null && testingSpec.quantizationConfig.enableADC) {
            float[] rawFloat = generateOneSingleFloatVector(
                DIMENSIONS,
                testingSpec.minValue,
                testingSpec.maxValue,
                spaceType == SpaceType.COSINESIMIL
            );

            (QuantizationService.getInstance()).transformWithADC(testingSpec.quantizationState, rawFloat, spaceType);

            query = queryForVectorReader = rawFloat;

            resultsFromFaiss = JNIService.queryIndex(
                indexPointer,
                (float[]) query,
                k,
                buildInfo.parameters,
                KNNEngine.FAISS,
                filteredIds,
                FilterIdsSelector.FilterIdsSelectorType.BATCH.getValue(),
                parentIds
            );
        } else if (testingSpec.dataType == VectorDataType.BINARY) {
            if (testingSpec.quantizationParams != null) {
                float[] floatQuery = generateOneSingleFloatVector(
                    DIMENSIONS,
                    testingSpec.minValue,
                    testingSpec.maxValue,
                    spaceType == SpaceType.COSINESIMIL
                );
                query = queryForVectorReader = QuantizationService.getInstance()
                    .quantize(
                        testingSpec.quantizationState,
                        floatQuery,
                        QuantizationService.getInstance().createQuantizationOutput(testingSpec.quantizationParams)
                    );
            } else {
                queryForVectorReader = generateOneSingleByteVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);
                query = queryForVectorReader;
            }

            resultsFromFaiss = JNIService.queryBinaryIndex(
                indexPointer,
                (byte[]) query,
                k,
                buildInfo.parameters,
                KNNEngine.FAISS,
                filteredIds,
                FilterIdsSelector.FilterIdsSelectorType.BATCH.getValue(),
                parentIds
            );
        } else {
            throw new AssertionError();
        }

        JNIService.free(indexPointer, KNNEngine.FAISS);

        // Score transform in Faiss (source of truth)
        for (int i = 0; i < resultsFromFaiss.length; i++) {
            resultsFromFaiss[i] = new KNNQueryResult(
                resultsFromFaiss[i].getId(),
                KNNEngine.FAISS.score(resultsFromFaiss[i].getScore(), spaceType)
            );
        }

        // Search via VectorReader
        final KNNQueryResult[] resultsFromVectorReader = doSearchViaVectorReader(
            buildInfo,
            queryForVectorReader,
            testingSpec.quantizationConfig != null && testingSpec.quantizationConfig.enableADC
                ? VectorDataType.FLOAT
                : testingSpec.dataType,
            filteredIds,
            k,
            doExhaustiveSearch,
            spaceType,
            testingSpec.directoryClass
        );

        // Validate results
        validateResults(
            buildInfo.documentIds,
            buildInfo.vectors,
            testingSpec.quantizationConfig != null && testingSpec.quantizationConfig.enableADC
                ? convertToFloatArray(
                    (byte[]) QuantizationService.getInstance()
                        .quantize(
                            testingSpec.quantizationState,
                            query,
                            QuantizationService.getInstance().createQuantizationOutput(testingSpec.quantizationParams)
                        )
                )
                : query,
            filteredIds,
            resultsFromFaiss,
            resultsFromVectorReader,
            spaceType.getKnnVectorSimilarityFunction(),
            TOP_K,
            testingSpec.quantizationConfig != null && testingSpec.quantizationConfig.enableADC
        );
    }

    @SneakyThrows
    private static <D extends Directory> KNNQueryResult[] doSearchViaVectorReader(
        BuildInfo buildInfo,
        Object query,
        VectorDataType vectorDataType,
        long[] filteredIds,
        final int k,
        final boolean exhaustiveSearch,
        final SpaceType spaceType,
        final Class<D> directoryClass
    ) {
        // Make KNN vector field info
        KNNCodecTestUtil.FieldInfoBuilder fieldInfoBuilder = KNNCodecTestUtil.FieldInfoBuilder.builder(TARGET_FIELD)
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());

        // Add space type from build parameters
        if (buildInfo.parameters.containsKey(SPACE_TYPE)) {
            fieldInfoBuilder.addAttribute(KNNConstants.SPACE_TYPE, (String) buildInfo.parameters.get(SPACE_TYPE));
        }

        final boolean isQuantizedIndex = buildInfo.parameters.containsKey(QFRAMEWORK_CONFIG);
        if (isQuantizedIndex) {
            fieldInfoBuilder.addAttribute(QFRAMEWORK_CONFIG, (String) buildInfo.parameters.get(QFRAMEWORK_CONFIG));
        }

        // This test uses float indices, so no special quantization config needed

        // TODO: pass in quantization config here.
        FieldInfo vectorField = fieldInfoBuilder.build();
        final FieldInfo[] vectorFieldArr = new FieldInfo[] { vectorField };
        final FieldInfos fieldInfos = new FieldInfos(vectorFieldArr);

        // Make segment info
        final SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.getUseCompoundFile()).thenReturn(false);
        when(segmentInfo.files()).thenReturn(Set.of(buildInfo.faissIndexFile));
        when(segmentInfo.getId()).thenReturn("LuceneOnFaiss".getBytes());
        when(segmentInfo.getVersion()).thenReturn(org.apache.lucene.util.Version.LATEST);

        // Prepare collector and bits
        // buildInfo.documentIds.size() + 1 -> Will force it to do exhaustive search.
        // buildInfo.documentIds.size() - 1 -> Will maximize search space, this is equivalent to set efSearch = len(N) - 1
        final int efSearch = exhaustiveSearch ? buildInfo.documentIds.size() + 1 : buildInfo.documentIds.size() - 1;
        final KnnCollector knnCollector = new TopKnnCollector(efSearch, Integer.MAX_VALUE);
        FixedBitSet fixedBitSet = null;
        if (filteredIds != null) {
            fixedBitSet = new FixedBitSet(buildInfo.documentIds.getLast() + 10);
            for (long filteredId : filteredIds) {
                fixedBitSet.set((int) filteredId);
            }
        }

        AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(fixedBitSet, buildInfo.documentIds.getLast() + 10);

        // Make SegmentReadState and do search
        try (final Directory rawDirectory = directoryClass.getConstructor(Path.class).newInstance(buildInfo.tempDirPath)) {
            // Write real flat vectors to the directory so we can create a real FlatVectorsReader
            final String segmentName = "_flat";

            // Create a single SegmentInfo for flat vectors (shared between write and read)
            final byte[] flatSegId = StringHelper.randomId();
            final SegmentInfo flatSegInfo = new SegmentInfo(
                rawDirectory,
                Version.LATEST,
                Version.LATEST,
                segmentName,
                buildInfo.documentIds.isEmpty() ? 0 : buildInfo.documentIds.getLast() + 1,
                false,
                false,
                null,
                Collections.emptyMap(),
                flatSegId,
                Collections.emptyMap(),
                null
            );

            // Create flat vector FieldInfo
            final FieldInfo flatFieldInfo = createFlatVectorFieldInfo(vectorField, vectorDataType, spaceType);
            final FieldInfos flatFieldInfos = new FieldInfos(new FieldInfo[] { flatFieldInfo });

            writeFlatVectors(rawDirectory, flatSegInfo, flatFieldInfos, flatFieldInfo, buildInfo, vectorDataType);

            // Wrap directory with spy to track file reads during warmup
            final ReadTrackingDirectory spyDirectory = new ReadTrackingDirectory(rawDirectory);

            // Create a real FlatVectorsReader from the written flat vectors
            final SegmentReadState flatReadState = new SegmentReadState(spyDirectory, flatSegInfo, flatFieldInfos, IOContext.DEFAULT);
            final FlatVectorsFormat flatFormat = new Lucene99FlatVectorsFormat(new PrefetchableFlatVectorScorer(SCORER));
            final FlatVectorsReader realFlatVectorsReader = flatFormat.fieldsReader(flatReadState);

            // Create the main SegmentReadState with the spy directory
            final SegmentReadState readState = new SegmentReadState(spyDirectory, segmentInfo, fieldInfos, IOContext.DEFAULT);

            try (NativeEngines990KnnVectorsReader vectorsReader = new NativeEngines990KnnVectorsReader(readState, realFlatVectorsReader)) {
                // Warmup: invoke warmup via WarmableReader before search
                // Reset read flags so we only track reads that happen during warmup
                spyDirectory.resetReadFlags();
                vectorsReader.warmUp(TARGET_FIELD);

                // Assert file read patterns based on field type
                assertTrue("Warmup should read .faiss file, but it was not read", spyDirectory.wasExtensionRead(".faiss"));
                if (isQuantizedIndex) {
                    // Feature: warmup-delegation-tests, Property 3: Qframe warmup reads .faiss and .vec
                    assertTrue("Warmup for qframe field should read .vec file, but it was not read", spyDirectory.wasExtensionRead(".vec"));
                } else {
                    // Feature: warmup-delegation-tests, Property 2: Non-qframe warmup reads only .faiss
                    if (spyDirectory.hasTrackedExtension(".vec")) {
                        assertFalse(
                            "Warmup for non-qframe field should NOT read .vec file, but it was read",
                            spyDirectory.wasExtensionRead(".vec")
                        );
                    }
                }

                // Proceed with normal search
                if (vectorDataType == VectorDataType.FLOAT) {
                    vectorsReader.search(TARGET_FIELD, (float[]) query, knnCollector, acceptDocs);
                } else if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
                    vectorsReader.search(TARGET_FIELD, (byte[]) query, knnCollector, acceptDocs);
                } else {
                    throw new AssertionError();
                }
            } finally {
                realFlatVectorsReader.close();
            }
        }

        // Make results
        final TopDocs topDocs = knnCollector.topDocs();
        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        if (spaceType == SpaceType.COSINESIMIL) {
            MemoryOptimizedSearchScoreConverter.convertToCosineScore(scoreDocs);
        }
        assertTrue(scoreDocs.length >= k);
        final List<KNNQueryResult> results = new ArrayList<>();
        for (int i = 0; i < k; ++i) {
            results.add(new KNNQueryResult(scoreDocs[i].doc, scoreDocs[i].score));
        }
        return results.toArray(new KNNQueryResult[0]);
    }

    /**
     * Creates a FieldInfo suitable for Lucene99FlatVectorsFormat.
     * Uses FLOAT32 encoding for float and binary types, BYTE encoding for byte type.
     */
    private static FieldInfo createFlatVectorFieldInfo(FieldInfo original, VectorDataType dataType, SpaceType spaceType) {
        final VectorEncoding encoding = (dataType == VectorDataType.BYTE) ? VectorEncoding.BYTE : VectorEncoding.FLOAT32;

        return new FieldInfo(
            original.getName(),
            original.number,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            original.attributes(),
            0,
            0,
            0,
            DIMENSIONS,
            encoding,
            // Since we don't use this function for scoring, and this is merely for warm-up testing (which just loads vectors)
            // We can hard code L2 in here
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
    }

    /**
     * Writes real vectors from BuildInfo to the directory using Lucene99FlatVectorsFormat so that a real
     * FlatVectorsReader can be created for warmup testing.
     *
     * - Float: writes the actual float vectors from BuildInfo.vectors.floatVectors
     * - Byte: writes the actual byte vectors from BuildInfo.vectors.byteVectors
     * - Binary: writes randomly generated float vectors (binary indices don't read .vec during warmup anyway)
     */
    @SneakyThrows
    private static void writeFlatVectors(
        Directory directory,
        SegmentInfo segInfo,
        FieldInfos flatFieldInfos,
        FieldInfo flatFieldInfo,
        BuildInfo buildInfo,
        VectorDataType dataType
    ) {
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            directory,
            segInfo,
            flatFieldInfos,
            null,
            IOContext.DEFAULT
        );

        final FlatVectorsFormat flatFormat = new Lucene99FlatVectorsFormat(new PrefetchableFlatVectorScorer(SCORER));

        final int numVectors = buildInfo.documentIds.size();
        final int maxDoc = buildInfo.documentIds.isEmpty() ? 0 : buildInfo.documentIds.getLast() + 1;

        try (FlatVectorsWriter flatWriter = flatFormat.fieldsWriter(writeState)) {
            if (dataType == VectorDataType.BYTE) {
                // Write actual byte vectors
                @SuppressWarnings("unchecked")
                KnnFieldVectorsWriter<byte[]> fieldWriter = (KnnFieldVectorsWriter<byte[]>) flatWriter.addField(flatFieldInfo);
                for (int i = 0; i < numVectors; i++) {
                    int docId = buildInfo.documentIds.get(i);
                    byte[] vec = buildInfo.vectors.byteVectors.get(docId);
                    if (vec != null) {
                        fieldWriter.addValue(docId, vec);
                    }
                }
            } else {
                final List<float[]> floatVectors;
                if (buildInfo.parameters.containsKey(QFRAMEWORK_CONFIG)) {
                    // Use original vectors if it's quantized
                    floatVectors = buildInfo.originalVectors.floatVectors;
                } else {
                    floatVectors = buildInfo.vectors.floatVectors;
                }

                // FP32, FP16 and quantized index
                // Write actual float vectors
                @SuppressWarnings("unchecked")
                KnnFieldVectorsWriter<float[]> fieldWriter = (KnnFieldVectorsWriter<float[]>) flatWriter.addField(flatFieldInfo);
                for (int i = 0; i < numVectors; i++) {
                    int docId = buildInfo.documentIds.get(i);
                    float[] vec = floatVectors.get(docId);
                    if (vec != null) {
                        fieldWriter.addValue(docId, vec);
                    }
                }
            }
            flatWriter.flush(maxDoc, null);
            flatWriter.finish();
        }
    }

    @SneakyThrows
    private BuildInfo buildFaissIndex(
        final TestingSpec testingSpec,
        final int numberOfTotalDocsInSegment,
        final IndexingType indexingType,
        final SpaceType spaceType
    ) {
        final Path tempDir = createTempDir(UUID.randomUUID().toString());
        final String fileName = UUID.randomUUID() + "_" + TARGET_FIELD + ".faiss";
        BuildInfo buildInfo = null;
        try (final Directory directory = newFSDirectory(tempDir)) {
            // Set up basic parameters
            try (final IndexOutput indexOutput = directory.createOutput(fileName, IOContext.DEFAULT)) {
                final BuildIndexParams.BuildIndexParamsBuilder builder = BuildIndexParams.builder();
                FieldInfo fieldInfo = mock(FieldInfo.class);
                when(fieldInfo.getName()).thenReturn(TARGET_FIELD);
                builder.field(fieldInfo.getName())
                    .knnEngine(KNNEngine.FAISS)
                    .vectorDataType(testingSpec.dataType)
                    .indexOutputWithBuffer(new IndexOutputWithBuffer(indexOutput));

                // Set up parameters
                final Map<String, Object> parameters = new HashMap<>();
                parameters.put(NAME, METHOD_HNSW);
                parameters.put(VECTOR_DATA_TYPE_FIELD, testingSpec.dataType.getValue());
                parameters.put(SPACE_TYPE, spaceType.getValue());
                parameters.put(INDEX_THREAD_QTY, 1);
                parameters.put(INDEX_DESCRIPTION_PARAMETER, testingSpec.indexDescription);

                final Map<String, Object> methodParameters = new HashMap<>();
                parameters.put(PARAMETERS, methodParameters);
                methodParameters.put(METHOD_PARAMETER_EF_SEARCH, numberOfTotalDocsInSegment - 1);
                methodParameters.put(METHOD_PARAMETER_EF_CONSTRUCTION, numberOfTotalDocsInSegment);

                methodParameters.put(METHOD_ENCODER_PARAMETER, testingSpec.encoderParameters);

                // unit test for ADC, bit of a shortcut so we can hand build the quantization config.
                if (testingSpec.quantizationConfig != null) {
                    parameters.put(QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(testingSpec.quantizationConfig));
                }

                builder.indexParameters(parameters);

                // Set up vectors
                final List<Integer> documentIds = indexingType.generateDocumentIds(numberOfTotalDocsInSegment);
                buildInfo = new BuildInfo(tempDir, fileName, parameters, documentIds);
                builder.totalLiveDocs(documentIds.size());

                if (testingSpec.dataType == VectorDataType.BYTE) {
                    final List<byte[]> vectors = generateRandomByteVectors(
                        documentIds,
                        DIMENSIONS,
                        testingSpec.minValue,
                        testingSpec.maxValue
                    );
                    buildInfo.vectors = new SearchTestHelper.Vectors(VectorDataType.BYTE, vectors);
                    final KNNVectorValues<byte[]> byteVectorValues = createKNNByteVectorValues(documentIds, vectors);
                    builder.knnVectorValuesSupplier(() -> byteVectorValues);
                } else if (testingSpec.dataType == VectorDataType.FLOAT) {
                    final List<float[]> floatVectors = SearchTestHelper.generateRandomFloatVectors(
                        buildInfo.documentIds,
                        DIMENSIONS,
                        testingSpec.minValue,
                        testingSpec.maxValue,
                        spaceType == SpaceType.COSINESIMIL
                    );
                    buildInfo.vectors = new SearchTestHelper.Vectors(floatVectors);
                    final KNNFloatVectorValues floatVectorValues = createKNNFloatVectorValues(documentIds, floatVectors);
                    builder.knnVectorValuesSupplier(() -> floatVectorValues);
                } else if (testingSpec.dataType == VectorDataType.BINARY) {
                    // Get random float vectors
                    final List<float[]> floatVectors = SearchTestHelper.generateRandomFloatVectors(
                        buildInfo.documentIds,
                        DIMENSIONS,
                        testingSpec.minValue,
                        testingSpec.maxValue,
                        false
                    );
                    assert (testingSpec.quantizationParams != null);

                    // Get quantization state
                    final QuantizationState quantizationState = QuantizationService.getInstance()
                        .train(
                            testingSpec.quantizationParams,
                            () -> (KNNVectorValues) createKNNFloatVectorValues(documentIds, floatVectors),
                            documentIds.size()
                        );
                    testingSpec.quantizationState = quantizationState;
                    builder.quantizationState(quantizationState);

                    // Set quantized vectors
                    final List<byte[]> quantizedVectors = floatVectors.stream().map(v -> {
                        if (v != null) {
                            return (byte[]) QuantizationService.getInstance()
                                .quantize(
                                    testingSpec.quantizationState,
                                    v,
                                    QuantizationService.getInstance().createQuantizationOutput(testingSpec.quantizationParams)
                                );
                        }

                        // For sparse case, it can have null value indicating not having a vector.
                        // Ex: [[..], [..], null, null, [..], ..], in which 2 and 3 docs don't have vectors.
                        return null;
                    }).toList();
                    buildInfo.vectors = new SearchTestHelper.Vectors(VectorDataType.BINARY, quantizedVectors);
                    buildInfo.originalVectors = new SearchTestHelper.Vectors(floatVectors);

                    // Set values supplier
                    // floatVectorValues is already exhausted, need to create a new one.
                    final KNNVectorValues floatVectorValuesForValidation = createKNNFloatVectorValues(documentIds, floatVectors);
                    builder.knnVectorValuesSupplier(() -> floatVectorValuesForValidation);
                } else {
                    throw new AssertionError();
                }

                // Now start indexing
                final BuildIndexParams buildIndexParams = builder.build();
                MemoryOptimizedSearchIndexingSupport.buildIndex(buildIndexParams);
            }
        }

        return buildInfo;
    }

    public static void validateResults(
        final List<Integer> documentIds,
        final SearchTestHelper.Vectors vectors,
        final Object query,
        final long[] filteredIds,
        KNNQueryResult[] resultsFromFaiss,
        KNNQueryResult[] resultsFromVectorReader,
        KNNVectorSimilarityFunction similarityFunction,
        final int topK,
        final boolean isAdc
    ) {
        final Set<Integer> answerDocIds = getKnnAnswerSetForVectors(documentIds, vectors, query, filteredIds, similarityFunction, topK);

        final Set<Integer> expectedDocIds = Arrays.stream(resultsFromFaiss)
            .mapToInt(KNNQueryResult::getId)
            .boxed()
            .collect(Collectors.toSet());
        int answerMatchCount = 0;
        int matchCount = 0;
        for (int i = 0; i < resultsFromFaiss.length; ++i) {
            if (expectedDocIds.contains(resultsFromVectorReader[i].getId())) {
                ++matchCount;
            }
            if (answerDocIds.contains(resultsFromVectorReader[i].getId())) {
                ++answerMatchCount;
            }
        }

        final float matchRatio = ((float) matchCount) / resultsFromFaiss.length;
        final float recall = ((float) answerMatchCount) / topK;

        // It can happen that match ratio between FAISS and MemOptimizedSearch is lower than 80%, but if it happens with a recall lower than
        // 0.8 indicates something's off. We use a smaller match threshold for ADC.
        if (isAdc) {
            assertFalse("matchRatio=" + matchRatio + " < 0.6 && recall=" + recall + " < 0.8", matchRatio < 0.6 && recall < 0.8);
        } else {
            assertFalse("matchRatio=" + matchRatio + " < 0.8 && recall=" + recall + " < 0.8", matchRatio < 0.8 && recall < 0.8);
        }

        // Validate score values are the same
        final Map<Integer, Float> faissIdScores = Arrays.stream(resultsFromFaiss)
            .collect(toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
        final boolean isRunningInWindows = System.getProperty("os.name").toLowerCase().contains("win");
        if (isRunningInWindows == false) {
            // For unknown reason, this assertion is only failing in Windows
            // Until root causing the issue, blocking assertion for Windows.
            for (final KNNQueryResult result : resultsFromVectorReader) {
                if (faissIdScores.containsKey(result.getId())) {
                    final float scoreFromReader = result.getScore();
                    final float faissScore = faissIdScores.get(result.getId());
                    assertEquals(faissScore, scoreFromReader, 1e-3);
                }
            }
        }
    }

    @SneakyThrows
    public static KNNFloatVectorValues createKNNFloatVectorValues(final List<Integer> documentIds, final List<float[]> vectors) {

        final KNNVectorValuesIterator iterator = new KNNVectorValuesIterator() {
            private int index = -1;

            @Override
            public int docId() {
                if (index == -1) {
                    return -1;
                } else if (index == DocIdSetIterator.NO_MORE_DOCS) {
                    return DocIdSetIterator.NO_MORE_DOCS;
                }
                return documentIds.get(index);
            }

            @Override
            public int advance(int docId) throws IOException {
                throw new UnsupportedEncodingException();
            }

            @Override
            public int nextDoc() {
                if ((index + 1) >= documentIds.size()) {
                    index = DocIdSetIterator.NO_MORE_DOCS;
                    return DocIdSetIterator.NO_MORE_DOCS;
                }

                return documentIds.get(++index);
            }

            @Override
            public DocIdSetIterator getDocIdSetIterator() {
                return null;
            }

            @Override
            public long liveDocs() {
                return documentIds.size();
            }

            @Override
            public VectorValueExtractorStrategy getVectorExtractorStrategy() {
                return new VectorValueExtractorStrategy() {
                    @Override
                    public float[] extract(VectorDataType vectorDataType, KNNVectorValuesIterator vectorValuesIterator) {
                        return vectors.get(vectorValuesIterator.docId());
                    }
                };
            }
        };

        // Instantiate KNNFloatVectorValues
        Constructor<KNNFloatVectorValues> constructor = KNNFloatVectorValues.class.getDeclaredConstructor(KNNVectorValuesIterator.class);
        constructor.setAccessible(true);
        return constructor.newInstance(iterator);
    }

    @SneakyThrows
    public static KNNByteVectorValues createKNNByteVectorValues(final List<Integer> documentIds, final List<byte[]> vectors) {
        final KNNVectorValuesIterator iterator = new KNNVectorValuesIterator() {
            private int index = -1;

            @Override
            public int docId() {
                if (index == -1) {
                    return -1;
                } else if (index == DocIdSetIterator.NO_MORE_DOCS) {
                    return DocIdSetIterator.NO_MORE_DOCS;
                }
                return documentIds.get(index);
            }

            @Override
            public int advance(int docId) throws IOException {
                throw new UnsupportedEncodingException();
            }

            @Override
            public int nextDoc() {
                if ((index + 1) >= documentIds.size()) {
                    index = DocIdSetIterator.NO_MORE_DOCS;
                    return DocIdSetIterator.NO_MORE_DOCS;
                }

                return documentIds.get(++index);
            }

            @Override
            public DocIdSetIterator getDocIdSetIterator() {
                return null;
            }

            @Override
            public long liveDocs() {
                return documentIds.size();
            }

            @Override
            public VectorValueExtractorStrategy getVectorExtractorStrategy() {
                return new VectorValueExtractorStrategy() {
                    @Override
                    public byte[] extract(VectorDataType vectorDataType, KNNVectorValuesIterator vectorValuesIterator) {
                        return vectors.get(vectorValuesIterator.docId());
                    }
                };
            }
        };

        // Instantiate KNNFloatVectorValues
        Constructor<KNNByteVectorValues> constructor = KNNByteVectorValues.class.getDeclaredConstructor(KNNVectorValuesIterator.class);
        constructor.setAccessible(true);
        return constructor.newInstance(iterator);
    }

    /**
     * When searching a CAGRA HNSW index with a non-seeded collector, the searcher should inject
     * RandomEntryPointsKnnSearchStrategy. This test verifies the search completes successfully
     * and returns valid results with the default (non-seeded) strategy.
     */
    @SneakyThrows
    public void testCagraSearch_whenNonSeededCollector_thenSearchSucceeds() {
        final int dimension = 768;
        final int totalVectors = 300;
        final int k = 100;

        final IndexInput input = FaissHNSWTests.loadHnswBinary("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin");
        FieldInfo fieldInfo = mock(FieldInfo.class);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(
            input,
            FaissIndex.load(input),
            fieldInfo,
            FlatVectorsScorerProvider.getFlatVectorsScorer(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, SCORER)
        );

        // Use a non-seeded strategy (default HNSW strategy)
        final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
        final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(null, totalVectors);

        // Build a random query
        final float[] query = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            query[i] = (float) Math.random();
        }

        // Search should succeed — RandomEntryPointsKnnSearchStrategy is used internally for CAGRA
        searcher.search(query, knnCollector, acceptDocs);
        final TopDocs topDocs = knnCollector.topDocs();
        assertTrue("Should return results for non-seeded CAGRA search", topDocs.scoreDocs.length > 0);
    }

    /**
     * When searching a CAGRA HNSW index with a collector that already has a Seeded strategy,
     * the searcher should honor the existing seeded entry points and NOT override them with
     * RandomEntryPointsKnnSearchStrategy. This test verifies the search completes successfully
     * using the provided seeded strategy.
     */
    @SneakyThrows
    public void testCagraSearch_whenSeededCollector_thenHonorsSeededStrategy() {
        final int dimension = 768;
        final int totalVectors = 300;
        final int k = 100;

        final IndexInput input = FaissHNSWTests.loadHnswBinary("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin");
        FieldInfo mockedFieldInfo = mock(FieldInfo.class);
        Mockito.when(mockedFieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(
            input,
            FaissIndex.load(input),
            mockedFieldInfo,
            FlatVectorsScorerProvider.getFlatVectorsScorer(mockedFieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, SCORER)
        );

        // Create a Seeded strategy with known seed entry points
        final int numSeeds = 5;
        final DocIdSetIterator seedDocs = new DocIdSetIterator() {
            private int current = -1;

            @Override
            public int docID() {
                return current;
            }

            @Override
            public int nextDoc() {
                current++;
                if (current >= numSeeds) {
                    return NO_MORE_DOCS;
                }
                // Use first few vector ordinals as seeds
                return current * 10;
            }

            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return numSeeds;
            }
        };

        final KnnSearchStrategy seededStrategy = new KnnSearchStrategy.Seeded(seedDocs, numSeeds, KnnSearchStrategy.Hnsw.DEFAULT);
        final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE, seededStrategy);
        final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(null, totalVectors);

        // Build a random query
        final float[] query = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            query[i] = (float) Math.random();
        }

        // Search should succeed — the seeded strategy should be honored, not replaced
        searcher.search(query, knnCollector, acceptDocs);
        final TopDocs topDocs = knnCollector.topDocs();
        assertTrue("Should return results for seeded CAGRA search", topDocs.scoreDocs.length > 0);
    }

    /**
     * Verifies that when a seeded collector is used on a CAGRA index, the search results
     * are still valid (reasonable recall) compared to exhaustive search, confirming the
     * seeded entry points are being properly used for graph traversal.
     */
    @SneakyThrows
    public void testCagraSearch_whenSeededCollector_thenResultsHaveReasonableRecall() {
        final int dimension = 768;
        final int totalVectors = 300;
        final int k = 30;

        final IndexInput input = FaissHNSWTests.loadHnswBinary("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin");

        // Build a random query
        final float[] query = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            query[i] = (float) Math.random();
        }

        // First, do exhaustive search to get ground truth
        FieldInfo mockedFieldInfo = mock(FieldInfo.class);
        Mockito.when(mockedFieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        final FaissMemoryOptimizedSearcher exhaustiveSearcher = new FaissMemoryOptimizedSearcher(
            input,
            FaissIndex.load(input),
            mockedFieldInfo,
            FlatVectorsScorerProvider.getFlatVectorsScorer(mockedFieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, SCORER)
        );
        final KnnCollector exhaustiveCollector = new TopKnnCollector(totalVectors, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
        final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(null, totalVectors);
        exhaustiveSearcher.search(query, exhaustiveCollector, acceptDocs);
        final TopDocs exhaustiveTopDocs = exhaustiveCollector.topDocs();
        final Set<Integer> groundTruth = Arrays.stream(exhaustiveTopDocs.scoreDocs)
            .limit(k)
            .mapToInt(sd -> sd.doc)
            .boxed()
            .collect(Collectors.toSet());

        // Now search with a seeded collector on a fresh searcher
        input.seek(0);
        final FaissMemoryOptimizedSearcher seededSearcher = new FaissMemoryOptimizedSearcher(
            input,
            FaissIndex.load(input),
            mockedFieldInfo,
            FlatVectorsScorerProvider.getFlatVectorsScorer(mockedFieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, SCORER)
        );

        final int numSeeds = 3;
        final DocIdSetIterator seedDocs = new DocIdSetIterator() {
            private int current = -1;

            @Override
            public int docID() {
                return current;
            }

            @Override
            public int nextDoc() {
                current++;
                if (current >= numSeeds) {
                    return NO_MORE_DOCS;
                }
                return current * 50;
            }

            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return numSeeds;
            }
        };

        final KnnSearchStrategy seededStrategy = new KnnSearchStrategy.Seeded(seedDocs, numSeeds, KnnSearchStrategy.Hnsw.DEFAULT);
        final KnnCollector seededCollector = new TopKnnCollector(k, Integer.MAX_VALUE, seededStrategy);
        seededSearcher.search(query, seededCollector, acceptDocs);
        final TopDocs seededTopDocs = seededCollector.topDocs();

        // Verify recall is reasonable
        int matchCount = 0;
        for (ScoreDoc sd : seededTopDocs.scoreDocs) {
            if (groundTruth.contains(sd.doc)) {
                matchCount++;
            }
        }
        final float recall = (float) matchCount / k;
        assertTrue("Seeded CAGRA search recall should be > 0.5, was " + recall, recall > 0.5);
    }

    // Feature: warmup-delegation-tests, Property 2: Non-qframe warmup reads only .faiss
    // Feature: warmup-delegation-tests, Property 3: Qframe warmup reads .faiss and .vec

    /**
     * An IndexInput wrapper that delegates all operations to the underlying IndexInput
     * and tracks whether any read method was called via a {@code wasRead} flag.
     * Used by the spy Directory to verify warmup file-read patterns.
     */
    static class ReadTrackingIndexInput extends IndexInput {
        private final IndexInput delegate;
        volatile boolean wasRead = false;

        ReadTrackingIndexInput(String resourceDescription, IndexInput delegate) {
            super(resourceDescription);
            this.delegate = delegate;
        }

        private void markRead() {
            wasRead = true;
        }

        @Override
        public byte readByte() throws IOException {
            markRead();
            return delegate.readByte();
        }

        @Override
        public void readBytes(byte[] b, int offset, int len) throws IOException {
            markRead();
            delegate.readBytes(b, offset, len);
        }

        @Override
        public void readFloats(float[] floats, int offset, int len) throws IOException {
            markRead();
            delegate.readFloats(floats, offset, len);
        }

        @Override
        public void readInts(int[] dst, int offset, int len) throws IOException {
            markRead();
            delegate.readInts(dst, offset, len);
        }

        @Override
        public void readLongs(long[] dst, int offset, int len) throws IOException {
            markRead();
            delegate.readLongs(dst, offset, len);
        }

        @Override
        public short readShort() throws IOException {
            markRead();
            return delegate.readShort();
        }

        @Override
        public int readInt() throws IOException {
            markRead();
            return delegate.readInt();
        }

        @Override
        public long readLong() throws IOException {
            markRead();
            return delegate.readLong();
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }

        @Override
        public long getFilePointer() {
            return delegate.getFilePointer();
        }

        @Override
        public void seek(long pos) throws IOException {
            delegate.seek(pos);
        }

        @Override
        public long length() {
            return delegate.length();
        }

        @Override
        public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
            return new ReadTrackingIndexInput(sliceDescription, delegate.slice(sliceDescription, offset, length)) {
                {
                    // Share the wasRead flag with the parent so any read on a slice is tracked
                    // We achieve this by overriding markRead to also set the parent's flag
                }

                @Override
                public byte readByte() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readByte();
                }

                @Override
                public void readBytes(byte[] b, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readBytes(b, off, len);
                }

                @Override
                public void readFloats(float[] floats, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readFloats(floats, off, len);
                }

                @Override
                public void readInts(int[] dst, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readInts(dst, off, len);
                }

                @Override
                public void readLongs(long[] dst, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readLongs(dst, off, len);
                }

                @Override
                public short readShort() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readShort();
                }

                @Override
                public int readInt() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readInt();
                }

                @Override
                public long readLong() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readLong();
                }
            };
        }

        @Override
        public IndexInput clone() {
            ReadTrackingIndexInput cloned = new ReadTrackingIndexInput(toString(), delegate.clone());
            // Share the wasRead flag: reads on the clone should mark the original as read
            return new ReadTrackingIndexInput(toString(), cloned.delegate) {
                @Override
                public byte readByte() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readByte();
                }

                @Override
                public void readBytes(byte[] b, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readBytes(b, off, len);
                }

                @Override
                public void readFloats(float[] floats, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readFloats(floats, off, len);
                }

                @Override
                public void readInts(int[] dst, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readInts(dst, off, len);
                }

                @Override
                public void readLongs(long[] dst, int off, int len) throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    super.readLongs(dst, off, len);
                }

                @Override
                public short readShort() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readShort();
                }

                @Override
                public int readInt() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readInt();
                }

                @Override
                public long readLong() throws IOException {
                    ReadTrackingIndexInput.this.markRead();
                    return super.readLong();
                }

                @Override
                public IndexInput clone() {
                    ReadTrackingIndexInput.this.markRead();
                    return ReadTrackingIndexInput.this.clone();
                }

                @Override
                public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
                    return ReadTrackingIndexInput.this.slice(sliceDescription, offset, length);
                }
            };
        }

        @Override
        public void prefetch(long offset, long length) throws IOException {
            delegate.prefetch(offset, length);
        }
    }

    /**
     * A spy Directory wrapper that intercepts {@code openInput()} calls, wraps returned
     * {@link IndexInput} objects with {@link ReadTrackingIndexInput}, and stores them in a
     * map keyed by file name. Only tracks files with extensions {@code .faiss}, {@code .vec},
     * or {@code .veb}.
     */
    static class ReadTrackingDirectory extends Directory {
        private static final Set<String> TRACKED_EXTENSIONS = Set.of(".faiss", ".vec", ".veb", ".veq");
        private final Directory delegate;
        final Map<String, ReadTrackingIndexInput> trackedInputs = new ConcurrentHashMap<>();

        ReadTrackingDirectory(Directory delegate) {
            this.delegate = delegate;
        }

        private static boolean shouldTrack(String name) {
            for (String ext : TRACKED_EXTENSIONS) {
                if (name.endsWith(ext)) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public IndexInput openInput(String name, IOContext context) throws IOException {
            IndexInput input = delegate.openInput(name, context);
            if (shouldTrack(name)) {
                ReadTrackingIndexInput tracked = new ReadTrackingIndexInput(name, input);
                trackedInputs.put(name, tracked);
                return tracked;
            }
            return input;
        }

        @Override
        public String[] listAll() throws IOException {
            return delegate.listAll();
        }

        @Override
        public void deleteFile(String name) throws IOException {
            delegate.deleteFile(name);
        }

        @Override
        public long fileLength(String name) throws IOException {
            return delegate.fileLength(name);
        }

        @Override
        public IndexOutput createOutput(String name, IOContext context) throws IOException {
            return delegate.createOutput(name, context);
        }

        @Override
        public IndexOutput createTempOutput(String prefix, String suffix, IOContext context) throws IOException {
            return delegate.createTempOutput(prefix, suffix, context);
        }

        @Override
        public void sync(java.util.Collection<String> names) throws IOException {
            delegate.sync(names);
        }

        @Override
        public void syncMetaData() throws IOException {
            delegate.syncMetaData();
        }

        @Override
        public void rename(String source, String dest) throws IOException {
            delegate.rename(source, dest);
        }

        @Override
        public java.util.Set<String> getPendingDeletions() throws IOException {
            return delegate.getPendingDeletions();
        }

        @Override
        public org.apache.lucene.store.Lock obtainLock(String name) throws IOException {
            return delegate.obtainLock(name);
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }

        /**
         * Returns whether the tracked file with the given extension was read.
         * Searches all tracked inputs for a file name ending with the given extension.
         */
        boolean wasExtensionRead(String extension) {
            for (Map.Entry<String, ReadTrackingIndexInput> entry : trackedInputs.entrySet()) {
                if (entry.getKey().endsWith(extension) && entry.getValue().wasRead) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Returns whether a tracked file with the given extension exists in the tracked inputs.
         */
        boolean hasTrackedExtension(String extension) {
            for (String name : trackedInputs.keySet()) {
                if (name.endsWith(extension)) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Resets all read tracking flags without clearing the tracked inputs map.
         * This allows tracking reads that happen after this point while keeping
         * the same IndexInput wrappers.
         */
        void resetReadFlags() {
            for (ReadTrackingIndexInput input : trackedInputs.values()) {
                input.wasRead = false;
            }
        }
    }

    @RequiredArgsConstructor
    static class BuildInfo {
        final Path tempDirPath;
        final String faissIndexFile;
        final Map<String, Object> parameters;
        final List<Integer> documentIds;
        SearchTestHelper.Vectors vectors;
        // This build might involve quantization having original vectors and resulting compressed vectors in `vectors` member variable.
        SearchTestHelper.Vectors originalVectors;
    }

    @RequiredArgsConstructor
    private static class TestingSpec {
        public final VectorDataType dataType;
        public final String indexDescription;
        public final float minValue;
        public final float maxValue;
        public final Map<String, Object> encoderParameters;
        public ScalarQuantizationParams quantizationParams;
        public QuantizationState quantizationState;
        public QuantizationConfig quantizationConfig;
        public Class directoryClass = MMapDirectory.class;
    }
}
