/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.AllArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.nativeindex.MemoryOptimizedSearchIndexingSupport;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
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
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QTY;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.generate.SearchTestHelper.generateOneSingleByteVector;
import static org.opensearch.knn.generate.SearchTestHelper.generateOneSingleFloatVector;
import static org.opensearch.knn.generate.SearchTestHelper.generateRandomByteVectors;
import static org.opensearch.knn.generate.SearchTestHelper.getKnnAnswerSetForVectors;

public class FaissMemoryOptimizedSearcherTests extends KNNTestCase {
    private static final String TARGET_FIELD = "target_field";
    private static final String FLOAT_HNSW_INDEX_DESCRIPTION = "HNSW16,Flat";
    private static final String BYTE_HNSW_INDEX_DESCRIPTION = "HNSW16,SQ8_direct_signed";
    private static final String FLOAT16_HNSW_INDEX_DESCRIPTION = "HNSW8,SQfp16";
    private static final int DIMENSIONS = 128;
    private static final int TOTAL_NUM_DOCS_IN_SEGMENT = 300;
    private static final int TOP_K = 30;
    private static final float NO_FILTERING = Float.NaN;

    public void testFloatIndexType() {
        final TestingSpec testingSpec = new TestingSpec(VectorDataType.FLOAT, FLOAT_HNSW_INDEX_DESCRIPTION, -1000000, 1000000);

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
        final TestingSpec testingSpec = new TestingSpec(VectorDataType.BYTE, BYTE_HNSW_INDEX_DESCRIPTION, -128, 127) {
            @Override
            public float trimValue(float value) {
                return (byte) value;
            }
        };

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
        final TestingSpec testingSpec = new TestingSpec(VectorDataType.FLOAT, FLOAT16_HNSW_INDEX_DESCRIPTION, -65504, 65504);

        // Test a dense case where all docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(testingSpec, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(testingSpec, IndexingType.DENSE_NESTED);
    }

    @SneakyThrows
    private void doSearchTest(final TestingSpec testingSpec, final IndexingType indexingType) {
        for (final SpaceType spaceType : Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT)) {
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
                indexPointer = JNIService.loadIndex(indexInputWithBuffer, buildInfo.parameters, KNNEngine.FAISS);
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
        final Object queryForVectorReader;
        final float[] query;
        final byte[] byteQuery;
        if (testingSpec.dataType == VectorDataType.FLOAT) {
            queryForVectorReader = query = generateOneSingleFloatVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);
        } else if (testingSpec.dataType == VectorDataType.BYTE) {
            queryForVectorReader = byteQuery = generateOneSingleByteVector(DIMENSIONS, testingSpec.minValue, testingSpec.maxValue);
            query = SearchTestHelper.convertToFloatArray(byteQuery);
        } else {
            throw new AssertionError();
        }

        final KNNQueryResult[] resultsFromFaiss = JNIService.queryIndex(
            indexPointer,
            query,
            k,
            buildInfo.parameters,
            KNNEngine.FAISS,
            filteredIds,
            FilterIdsSelector.FilterIdsSelectorType.BATCH.getValue(),
            parentIds
        );

        JNIService.free(indexPointer, KNNEngine.FAISS);

        // Search via VectorReader
        final KNNQueryResult[] resultsFromVectorReader = doSearchViaVectorReader(
            buildInfo,
            queryForVectorReader,
            testingSpec.dataType,
            filteredIds,
            k,
            doExhaustiveSearch
        );

        // Validate results
        validateResults(
            buildInfo.documentIds,
            buildInfo.vectors,
            query,
            filteredIds,
            resultsFromFaiss,
            resultsFromVectorReader,
            spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction(),
            TOP_K
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
        FieldInfo vectorField = KNNCodecTestUtil.FieldInfoBuilder.builder(TARGET_FIELD)
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .build();
        final FieldInfo[] vectorFieldArr = new FieldInfo[] { vectorField };
        final FieldInfos fieldInfos = new FieldInfos(vectorFieldArr);

        // Make segment info
        final SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.getUseCompoundFile()).thenReturn(false);
        when(segmentInfo.files()).thenReturn(Set.of(buildInfo.faissIndexFile));
        when(segmentInfo.getId()).thenReturn("LuceneOnFaiss".getBytes());

        // Prepare collector and bits
        // buildInfo.documentIds.size() + 1 -> Will force it to do exhaustive search.
        // buildInfo.documentIds.size() - 1 -> Will maximize search space, this is equivalent to set efSearch = len(N) - 1
        final int efSearch = exhaustiveSearch ? buildInfo.documentIds.size() + 1 : buildInfo.documentIds.size() - 1;
        final KnnCollector knnCollector = new TopKnnCollector(efSearch, Integer.MAX_VALUE);
        FixedBitSet acceptDocs = null;
        if (filteredIds != null) {
            acceptDocs = new FixedBitSet(buildInfo.documentIds.getLast() + 10);
            for (long filteredId : filteredIds) {
                acceptDocs.set((int) filteredId);
            }
        }

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
                } else if (vectorDataType == VectorDataType.BYTE) {
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
                methodParameters.put(METHOD_ENCODER_PARAMETER, Map.of(NAME, ENCODER_FLAT));
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
        final float[] query,
        final long[] filteredIds,
        KNNQueryResult[] resultsFromFaiss,
        KNNQueryResult[] resultsFromVectorReader,
        VectorSimilarityFunction similarityFunction,
        final int topK
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
        // 0.8 indicates something's off.
        assertFalse(matchRatio < 0.8 && recall < 0.8);
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
    @AllArgsConstructor
    private static class TestingSpec {
        public final VectorDataType dataType;
        public final String indexDescription;
        public float minValue;
        public float maxValue;

        public float trimValue(final float value) {
            return value;
        }
    }
}
