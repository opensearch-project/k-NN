/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

public class FaissMemoryOptimizedSearcherTests extends KNNTestCase {
    private static final String TARGET_FIELD = "target_field";
    private static final String FLOAT_HNSW_INDEX_DESCRIPTION = "HNSW16,Flat";
    private static final String BYTE_HNSW_INDEX_DESCRIPTION = "HNSW16,SQ8_direct_signed";
    private static final int DIMENSIONS = 128;
    private static final int MIN_VALUE = -1000000;
    private static final int MAX_VALUE = 1000000;
    private static final int TOTAL_NUM_DOCS_IN_SEGMENT = 1000;
    private static final int NUM_CHILD_DOCS = 5;

    public void testFloatIndexType() {
        // Test a dense case where all docs have KNN field.
        doSearchTest(VectorDataType.FLOAT, IndexingType.DENSE);

        // Test a sparse case where some docs don't have KNN field
        doSearchTest(VectorDataType.FLOAT, IndexingType.SPARSE);

        // Test a sparse nested case where some parent docs don't have KNN field
        doSearchTest(VectorDataType.FLOAT, IndexingType.SPARSE_NESTED);

        // Test a dense nested case where ALL parent docs have KNN field.
        doSearchTest(VectorDataType.FLOAT, IndexingType.DENSE_NESTED);
    }

    public void testByteIndexType() {
        // TODO(KDY) : Will be covered in part-6 (FP16 support)
        // doSearchTest(VectorDataType.BYTE, IndexingType.DENSE);
        // doSearchTest(VectorDataType.BYTE, IndexingType.SPARSE);
        // doSearchTest(VectorDataType.BYTE, IndexingType.SPARSE_NESTED);
        // doSearchTest(VectorDataType.BYTE, IndexingType.DENSE_NESTED);
    }

    @SneakyThrows
    private void doSearchTest(final VectorDataType dataType, final IndexingType indexingType) {
        doSearchTest(dataType, indexingType, false, false);
        doSearchTest(dataType, indexingType, false, true);
        doSearchTest(dataType, indexingType, true, false);
        doSearchTest(dataType, indexingType, true, false);
    }

    @SneakyThrows
    private void doSearchTest(
        final VectorDataType dataType,
        final IndexingType indexingType,
        final boolean doExhaustiveSearch,
        final boolean applyFiltering
    ) {
        // Build FAISS index
        final String indexDesc = dataType == VectorDataType.FLOAT ? FLOAT_HNSW_INDEX_DESCRIPTION : BYTE_HNSW_INDEX_DESCRIPTION;
        final BuildInfo buildInfo = buildFaissIndex(dataType, indexDesc, TOTAL_NUM_DOCS_IN_SEGMENT, IndexingType.DENSE);

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
        if (applyFiltering) {
            // Take only 80%. e.g. filtering 20% out
            final List<Integer> filteredDocIds = takePortions(buildInfo.documentIds, 0.8);
            filteredIds = filteredDocIds.stream().mapToLong(Integer::longValue).toArray();
        }

        // Reconstruct parent ids if it's necessary
        int[] parentIds = null;
        if (indexingType.isNested()) {
            parentIds = extractParentIds(buildInfo.documentIds);
        }

        // Take top-20 results
        final int k = 20;

        // Start search via JNI
        final Object queryForVectorReader;
        final float[] query;
        final byte[] byteQuery;
        if (dataType == VectorDataType.FLOAT) {
            queryForVectorReader = query = generateOneSingleFloatVector();
        } else if (dataType == VectorDataType.BYTE) {
            queryForVectorReader = byteQuery = generateOneSingleByteVector();
            query = new float[byteQuery.length];
            for (int i = 0; i < byteQuery.length; i++) {
                query[i] = byteQuery[i];
            }
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

        // Search via VectorReader
        final KNNQueryResult[] resultsFromVectorReader = doSearchViaVectorReader(
            buildInfo,
            queryForVectorReader,
            dataType,
            filteredIds,
            k,
            doExhaustiveSearch
        );

        // Validate results
        validateResults(resultsFromFaiss, resultsFromVectorReader);
    }

    private void validateResults(KNNQueryResult[] resultsFromFaiss, KNNQueryResult[] resultsFromVectorReader) {
        Set<Integer> expectedDocIds = new HashSet<>();
        for (int i = 0; i < resultsFromFaiss.length; ++i) {
            expectedDocIds.add(resultsFromFaiss[i].getId());
        }
        int matchCount = 0;
        for (int i = 0; i < resultsFromFaiss.length; ++i) {
            if (expectedDocIds.contains(resultsFromVectorReader[i].getId())) {
                ++matchCount;
            }
        }

        final float matchRatio = ((float) matchCount) / resultsFromFaiss.length;
        // Should match at least 80% with the one obtained from FAISS
        assertTrue(matchRatio >= 0.8);
    }

    @SneakyThrows
    private static KNNQueryResult[] doSearchViaVectorReader(
        BuildInfo buildInfo,
        Object query,
        VectorDataType vectorDataType,
        long[] filteredIds,
        int k,
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
        final int efSearch = exhaustiveSearch ? buildInfo.documentIds.size() + 1 : buildInfo.documentIds.size() - 1;
        final KnnCollector knnCollector = new TopKnnCollector(efSearch, Integer.MAX_VALUE);
        FixedBitSet acceptDocs = null;
        if (filteredIds != null) {
            acceptDocs = new FixedBitSet((int) filteredIds[filteredIds.length - 1] + 10);
            for (long filteredId : filteredIds) {
                acceptDocs.set((int) filteredId);
            }
        }

        // Make SegmentReadState and do search
        try (final Directory directory = newFSDirectory(buildInfo.tempDirPath)) {
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
        final VectorDataType dataType,
        final String indexDescription,
        final int numberOfTotalDocsInSegment,
        final IndexingType indexingType
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
                    .vectorDataType(dataType)
                    .indexOutputWithBuffer(new IndexOutputWithBuffer(indexOutput));

                // Set up parameters
                final Map<String, Object> parameters = new HashMap<>();
                parameters.put(NAME, METHOD_HNSW);
                parameters.put(VECTOR_DATA_TYPE_FIELD, dataType.getValue());
                parameters.put(SPACE_TYPE, SpaceType.L2.getValue());
                parameters.put(INDEX_THREAD_QTY, 1);
                parameters.put(INDEX_DESCRIPTION_PARAMETER, indexDescription);

                final Map<String, Object> methodParameters = new HashMap<>();
                parameters.put(PARAMETERS, methodParameters);
                methodParameters.put(METHOD_PARAMETER_EF_SEARCH, 100);
                methodParameters.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);
                methodParameters.put(METHOD_ENCODER_PARAMETER, Map.of(NAME, ENCODER_FLAT));
                builder.parameters(parameters);

                // Set up vectors
                final List<Integer> documentIds = indexingType.generateDocumentIds(numberOfTotalDocsInSegment);
                buildInfo = new BuildInfo(tempDir, fileName, parameters, documentIds);
                builder.totalLiveDocs(documentIds.size());

                if (dataType == VectorDataType.BYTE) {
                    final KNNVectorValues<byte[]> byteVectorValues = createKNNByteVectorValues(documentIds);
                    builder.knnVectorValuesSupplier(() -> byteVectorValues);
                } else if (dataType == VectorDataType.FLOAT) {
                    final KNNVectorValues<float[]> floatVectorValues = createKNNFloatVectorValues(documentIds);
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

    @SneakyThrows
    private KNNByteVectorValues createKNNByteVectorValues(final List<Integer> documentIds) {
        final List<byte[]> vectors = generateRandomByteVectors(documentIds);

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

    @SneakyThrows
    private static KNNFloatVectorValues createKNNFloatVectorValues(final List<Integer> documentIds) {
        final List<float[]> vectors = generateRandomFloatVectors(documentIds);

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

    private static float[] generateOneSingleFloatVector() {
        final float[] vector = new float[DIMENSIONS];
        for (int k = 0; k < DIMENSIONS; k++) {
            vector[k] = MIN_VALUE + ThreadLocalRandom.current().nextFloat() * (MAX_VALUE - MIN_VALUE);
        }
        return vector;
    }

    private static byte[] generateOneSingleByteVector() {
        final byte[] vector = new byte[DIMENSIONS];
        for (int k = 0; k < DIMENSIONS; k++) {
            vector[k] = (byte) (MIN_VALUE + ThreadLocalRandom.current().nextInt(MAX_VALUE - MIN_VALUE + 1));
        }
        return vector;
    }

    private static List<float[]> generateRandomFloatVectors(List<Integer> docIds) {
        final List<float[]> vectors = new ArrayList<>();
        for (int i = 0, j = 0; j < docIds.size(); j++) {
            // Add null vectors.
            // e.g. previous doc=15, current doc=18.
            // then put two nulls for doc=16, 17. This indicates that doc=16 and 17 don't have vector field.
            while (i < docIds.get(j)) {
                vectors.add(null);
                ++i;
            }

            vectors.add(generateOneSingleFloatVector());
            ++i;
        }

        return vectors;
    }

    private List<byte[]> generateRandomByteVectors(List<Integer> docIds) {
        final List<byte[]> vectors = new ArrayList<>();

        for (int i = 0, j = 0; j < docIds.size(); j++) {
            // Add null vectors.
            // e.g. previous doc=15, current doc=18.
            // then put two nulls for doc=16, 17. This indicates that doc=16 and 17 don't have vector field.
            while (i < docIds.get(j)) {
                vectors.add(null);
                ++i;
            }

            vectors.add(generateOneSingleByteVector());
            ++i;
        }

        return vectors;
    }

    private enum IndexingType {
        DENSE {
            @Override
            public List<Integer> generateDocumentIds(int numberOfTotalDocsInSegment) {
                return IntStream.rangeClosed(0, numberOfTotalDocsInSegment - 1).boxed().collect(Collectors.toList());
            }
        },
        SPARSE {
            @Override
            public List<Integer> generateDocumentIds(int numberOfTotalDocsInSegment) {
                List<Integer> docIds = new ArrayList<>(
                    IntStream.rangeClosed(0, numberOfTotalDocsInSegment - 1).boxed().collect(Collectors.toList())
                );

                // Take only 80% docs. e.g. 20% docs won't have vector field.
                return takePortions(docIds, 0.8);
            }
        },
        DENSE_NESTED {
            @Override
            public List<Integer> generateDocumentIds(int numberOfParentDocs) {
                final int numDocsHavingVector = NUM_CHILD_DOCS * numberOfParentDocs;
                final List<Integer> docIds = new ArrayList<>(numDocsHavingVector);
                for (int i = 0; i < numberOfParentDocs; i++) {
                    for (int j = 0; j < NUM_CHILD_DOCS; ++j) {
                        docIds.add(i * (NUM_CHILD_DOCS + 1) + j);
                    }
                }
                // Ex: [[0, 1, 2, 3, 4],
                // [6, 7, 8, 9, 10],
                // [12, ...]
                // Note that doc=5, doc=11 are parent document.
                return docIds;
            }

            @Override
            public boolean isNested() {
                return true;
            }
        },
        SPARSE_NESTED {
            @Override
            public List<Integer> generateDocumentIds(int numberOfTotalDocsInSegment) {
                // Ex: [[0, 1, 2, 3, 4],
                // [12, 13, 14, 15, 16],
                // [18, ...]
                // Note that docs in [6, 11] don't have vector fields.
                List<Integer> docIds = new ArrayList<>();
                int nextDocId = 0;
                for (int i = 0; i < numberOfTotalDocsInSegment; i++) {
                    // Only take 80%, or if it's last index
                    // Why force it to add child docs at the last index?
                    // -> So that we can always restore parent ids deterministically.
                    if (i == (numberOfTotalDocsInSegment - 1) || ThreadLocalRandom.current().nextFloat(1f) >= 0.8f) {
                        // This doc has vector
                        for (int j = 0; j < NUM_CHILD_DOCS; j++) {
                            docIds.add(nextDocId++);
                        }

                        // Visit a parent doc
                        ++nextDocId;
                    } else {
                        // This doc don't have vector
                        ++nextDocId;
                    }
                }

                return docIds;
            }

            @Override
            public boolean isNested() {
                return true;
            }
        };

        public abstract List<Integer> generateDocumentIds(int numberOfTotalDocsInSegment);

        public boolean isNested() {
            return false;
        }
    }

    @AllArgsConstructor
    static class BuildInfo {
        Path tempDirPath;
        String faissIndexFile;
        Map<String, Object> parameters;
        List<Integer> documentIds;
    }

    private static List<Integer> takePortions(final List<Integer> sequence, final double percentage) {
        List<Integer> newSequence = new ArrayList<>(sequence);
        Collections.shuffle(newSequence);
        newSequence = newSequence.subList(0, (int) (newSequence.size() * percentage));
        Collections.sort(newSequence);
        return newSequence;
    }

    private static int[] extractParentIds(final List<Integer> childDocIds) {
        // Reconstruct parent ids.
        // e.g. with [0,1,2,3,4, 6,7,8, 9,10,11,12,13], parent ids=[5, 14] with 5 child docs
        final List<Integer> parentIds = new ArrayList<>();
        for (int i = 0, j = 0; i < childDocIds.size(); ++i, ++j) {
            if (childDocIds.get(i) != j) {
                parentIds.add(j);
                j = i;
            }
        }

        // We always add child docs at the end.
        // So we can safely add parent id here.
        // Ex: With [, ..., 100, 101, 102, 103, 104], parent id would be 105 (e.g. having 5 child docs)
        parentIds.add(childDocIds.get(childDocIds.size() - 1) + 1);
        return parentIds.stream().mapToInt(Integer::intValue).toArray();
    }
}
