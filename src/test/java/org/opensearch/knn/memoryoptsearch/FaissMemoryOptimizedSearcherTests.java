/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
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
import org.apache.lucene.util.FixedBitSet;
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
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.query.FilterIdsSelector;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.index.vectorvalues.VectorValueExtractorStrategy;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Constructor;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

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

    public void test32xQuantizedBinaryIndexType() {
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.BINARY,
            BINARY_HSNW_INDEX_DESCRIPTION,
            -1000000,
            1000000,
            FLOAT32_ENCODER_PARAMETERS
        );
        testingSpec.quantizationParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();

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
            -1000000,
            1000000,
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
        final TestingSpec testingSpec = new TestingSpec(
            VectorDataType.FLOAT,
            FLOAT16_HNSW_INDEX_DESCRIPTION,
            -65504,
            65504,
            FLOAT16_ENCODER_PARAMETERS
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

    public void testADCWithBinaryQuantization() {
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

        adcEnabledSpec.isAdcEnabled = true;
        // Define parameter options
        List<IndexingType> indexingTypes = Arrays.asList(
            IndexingType.DENSE,
            IndexingType.DENSE_NESTED,
            IndexingType.SPARSE,
            IndexingType.SPARSE_NESTED
        );

        List<SpaceType> spaceTypes = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);

        List<Boolean> booleanOptions = Arrays.asList(true, false);

        List<Object> filterOptions = Arrays.asList(0.8f, NO_FILTERING);

        // // Generate cartesian product and run tests
        for (SpaceType spaceType : spaceTypes) {
            for (IndexingType indexingType : indexingTypes) {
                for (Boolean boolOption : booleanOptions) {
                    for (Object filterOption : filterOptions) {
                        doSearchTest(adcEnabledSpec, indexingType, spaceType, boolOption, (float) filterOption);
                    }
                }
            }
        }
    }

    @SneakyThrows
    private void doSearchTest(final TestingSpec testingSpec, final IndexingType indexingType) {
        final List<SpaceType> spaceTypes;
        if (testingSpec.dataType != VectorDataType.BINARY) {
            spaceTypes = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT);
        } else {
            spaceTypes = Arrays.asList(SpaceType.HAMMING);
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
                if (testingSpec.isAdcEnabled) {
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
                queryForVectorReader = query = generateOneSingleFloatVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);
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
        } else if (testingSpec.isAdcEnabled) {
            float[] rawFloat = (float[]) generateOneSingleFloatVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);

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
                float[] floatQuery = generateOneSingleFloatVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);
                query = queryForVectorReader = (byte[]) QuantizationService.getInstance()
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

        // Search via VectorReader
        final KNNQueryResult[] resultsFromVectorReader = doSearchViaVectorReader(
            buildInfo,
            queryForVectorReader,
            testingSpec.isAdcEnabled ? VectorDataType.FLOAT : testingSpec.dataType,
            filteredIds,
            k,
            doExhaustiveSearch
        );

        // Validate results
        validateResults(
            buildInfo.documentIds,
            buildInfo.vectors,
            testingSpec.isAdcEnabled
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
            testingSpec.isAdcEnabled
        );
    }

    @SneakyThrows
    private static KNNQueryResult[] doSearchViaVectorReader(
        BuildInfo buildInfo,
        Object query,
        VectorDataType vectorDataType,
        long[] filteredIds,
        final int k,
        final boolean exhaustiveSearch
    ) {
        // Make KNN vector field info
        KNNCodecTestUtil.FieldInfoBuilder fieldInfoBuilder = KNNCodecTestUtil.FieldInfoBuilder.builder(TARGET_FIELD)
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());

        // Add space type from build parameters
        if (buildInfo.parameters.containsKey(SPACE_TYPE)) {
            fieldInfoBuilder.addAttribute(KNNConstants.SPACE_TYPE, (String) buildInfo.parameters.get(SPACE_TYPE));
        }

        if (buildInfo.parameters.containsKey(QFRAMEWORK_CONFIG)) {
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
        final KnnCollector knnCollector = new TopKnnCollector(efSearch, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
        FixedBitSet fixedBitSet = null;
        if (filteredIds != null) {
            fixedBitSet = new FixedBitSet(buildInfo.documentIds.getLast() + 10);
            for (long filteredId : filteredIds) {
                fixedBitSet.set((int) filteredId);
            }
        }

        AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(fixedBitSet, buildInfo.documentIds.getLast() + 10);

        // Make SegmentReadState and do search
        try (final Directory directory = new MMapDirectory(buildInfo.tempDirPath)) {
            final SegmentReadState readState = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);
            try (
                NativeEngines990KnnVectorsReader vectorsReader = new NativeEngines990KnnVectorsReader(
                    readState,
                    mock(FlatVectorsReader.class)
                )
            ) {
                if (vectorDataType == VectorDataType.FLOAT) {
                    vectorsReader.search(TARGET_FIELD, (float[]) query, knnCollector, acceptDocs);
                } else if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
                    vectorsReader.search(TARGET_FIELD, (byte[]) query, knnCollector, acceptDocs);
                } else {
                    throw new AssertionError();
                }
            }
        }

        // Make results
        final TopDocs topDocs = knnCollector.topDocs();
        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        assertTrue(scoreDocs.length >= k);
        final List<KNNQueryResult> results = new ArrayList<>();
        for (int i = 0; i < k; ++i) {
            results.add(new KNNQueryResult(scoreDocs[i].doc, scoreDocs[i].score));
        }
        return results.toArray(new KNNQueryResult[0]);
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
                builder.fieldName(TARGET_FIELD)
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
                if (testingSpec.isAdcEnabled) {
                    parameters.put(QFRAMEWORK_CONFIG, "type=binary,bits=1,random_rotation=false,enable_adc=true");
                }

                builder.parameters(parameters);

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
                        testingSpec.maxValue
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
                        testingSpec.maxValue
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
            assertFalse(matchRatio < 0.6 && recall < 0.8);
        } else {
            assertFalse(matchRatio < 0.8 && recall < 0.8);
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

    @RequiredArgsConstructor
    static class BuildInfo {
        final Path tempDirPath;
        final String faissIndexFile;
        final Map<String, Object> parameters;
        final List<Integer> documentIds;
        SearchTestHelper.Vectors vectors;
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
        public boolean isAdcEnabled = false;
    }
}
