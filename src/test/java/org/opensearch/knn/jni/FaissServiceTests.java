/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.junit.BeforeClass;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;

public class FaissServiceTests extends KNNTestCase {

    static TestUtils.TestData testData;
    static TestUtils.TestData testDataNested;
    static String faissBinaryMethod = "BHNSW32";
    // k used by the ef_search tests, larger than the ef_search stored in the index.
    private static final int K = 200;
    private static final int INDEX_EF_SEARCH = 10;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (FaissServiceTests.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of FaissServiceTests Class is null");
        }
        URL testIndexVectors = JNIServiceTests.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testIndexVectorsNested = JNIServiceTests.class.getClassLoader().getResource("data/test_vectors_nested_1000x128.json");
        URL testQueries = JNIServiceTests.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testIndexVectorsNested != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
        testDataNested = new TestUtils.TestData(testIndexVectorsNested.getPath(), testQueries.getPath());
    }

    @SneakyThrows
    public void testLoadIndexWithStreamADC() {
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };

        for (SpaceType spaceType : spaceTypes) {
            Path tempDirPath = createTempDir();
            String indexFileName = "test_" + spaceType.getValue() + "_" + UUID.randomUUID() + ".tmp";

            try (Directory directory = newFSDirectory(tempDirPath)) {
                // Create an index with binary data
                long memoryAddr = testData.loadBinaryDataToMemoryAddress();
                TestUtils.createIndex(
                    testData.indexData.docs,
                    memoryAddr,
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName,
                    ImmutableMap.of(
                        INDEX_DESCRIPTION_PARAMETER,
                        faissBinaryMethod,
                        KNNConstants.SPACE_TYPE,
                        SpaceType.HAMMING.getValue(),
                        KNNConstants.VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BINARY.getValue()
                    ),
                    KNNEngine.FAISS
                );
                assertTrue(directory.fileLength(indexFileName) > 0);

                try (IndexInput indexInput = directory.openInput(indexFileName, IOContext.DEFAULT)) {
                    // Set up parameters for ADC loading
                    Map<String, Object> parameters = new HashMap<>();
                    parameters.put("space_type", spaceType.getValue());
                    // Use the exact string format expected by JNI util
                    parameters.put("quantization_level", "ScalarQuantizationParams_1");

                    long indexAddr = FaissService.loadIndexWithStreamADCParams(new IndexInputWithBuffer(indexInput), parameters);

                    // Test queries
                    for (float[] query : testData.queries) {
                        KNNQueryResult[] results = JNIService.queryIndex(
                            indexAddr,
                            query,
                            10,
                            Collections.emptyMap(),
                            KNNEngine.FAISS,
                            null,
                            0,
                            null
                        );
                        assertEquals(10, results.length);
                    }
                }
            }
        }
    }

    // Faiss sizes the HNSW candidate queue with max(efSearch, k) but stops exploring based on efSearch alone, so the
    // native engine must raise efSearch to k. See https://github.com/opensearch-project/k-NN/issues/3705
    @SneakyThrows
    public void testQueryIndex_whenKGreaterThanEfSearch_thenEfSearchRaisedToK() {
        try (Directory directory = newFSDirectory(createTempDir())) {
            final long pointer = createAndLoadHnswIndexWithSmallEfSearch(
                directory,
                "HNSW16,Flat",
                SpaceType.L2,
                VectorDataType.FLOAT,
                testData.loadDataToMemoryAddress()
            );
            try {
                for (float[] query : testData.queries) {
                    final int[] expectedIds = toIds(
                        JNIService.queryIndex(pointer, query, K, efSearchParameters(K), KNNEngine.FAISS, null, 0, null)
                    );
                    assertEquals(K, expectedIds.length);

                    // No query time ef_search: the ef_search stored in the index is raised to k.
                    assertArrayEquals(expectedIds, toIds(JNIService.queryIndex(pointer, query, K, null, KNNEngine.FAISS, null, 0, null)));
                    // Query time ef_search smaller than k is raised to k.
                    assertArrayEquals(
                        expectedIds,
                        toIds(JNIService.queryIndex(pointer, query, K, efSearchParameters(INDEX_EF_SEARCH), KNNEngine.FAISS, null, 0, null))
                    );
                }
            } finally {
                JNIService.free(pointer, KNNEngine.FAISS);
            }
        }
    }

    @SneakyThrows
    public void testQueryBinaryIndex_whenKGreaterThanEfSearch_thenEfSearchRaisedToK() {
        try (Directory directory = newFSDirectory(createTempDir())) {
            final long pointer = createAndLoadHnswIndexWithSmallEfSearch(
                directory,
                "BHNSW16",
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                testData.loadBinaryDataToMemoryAddress()
            );
            try {
                for (byte[] query : testData.binaryQueries) {
                    final int[] expectedIds = toIds(
                        JNIService.queryBinaryIndex(pointer, query, K, efSearchParameters(K), KNNEngine.FAISS, null, 0, null)
                    );
                    assertEquals(K, expectedIds.length);

                    // No method parameters, no filter and no parent ids: the ef_search stored in the index is raised to k.
                    assertArrayEquals(
                        expectedIds,
                        toIds(JNIService.queryBinaryIndex(pointer, query, K, null, KNNEngine.FAISS, null, 0, null))
                    );
                }
            } finally {
                JNIService.free(pointer, KNNEngine.FAISS, true);
            }
        }
    }

    @SneakyThrows
    private long createAndLoadHnswIndexWithSmallEfSearch(
        final Directory directory,
        final String indexDescription,
        final SpaceType spaceType,
        final VectorDataType vectorDataType,
        final long memoryAddress
    ) {
        final String indexFileName = "test_" + UUID.randomUUID() + ".tmp";
        TestUtils.createIndex(
            testData.indexData.docs,
            memoryAddress,
            testData.indexData.getDimension(),
            directory,
            indexFileName,
            ImmutableMap.of(
                INDEX_DESCRIPTION_PARAMETER,
                indexDescription,
                KNNConstants.SPACE_TYPE,
                spaceType.getValue(),
                KNNConstants.VECTOR_DATA_TYPE_FIELD,
                vectorDataType.getValue(),
                KNNConstants.PARAMETERS,
                ImmutableMap.of(KNNConstants.METHOD_PARAMETER_EF_SEARCH, INDEX_EF_SEARCH)
            ),
            KNNEngine.FAISS
        );
        try (IndexInput indexInput = directory.openInput(indexFileName, IOContext.DEFAULT)) {
            return JNIService.loadIndex(
                new IndexInputWithBuffer(indexInput),
                ImmutableMap.of(
                    INDEX_DESCRIPTION_PARAMETER,
                    indexDescription,
                    KNNConstants.VECTOR_DATA_TYPE_FIELD,
                    vectorDataType.getValue()
                ),
                KNNEngine.FAISS
            );
        }
    }

    private static Map<String, ?> efSearchParameters(final int efSearch) {
        return ImmutableMap.of(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearch);
    }

    private static int[] toIds(final KNNQueryResult[] results) {
        return Arrays.stream(results).mapToInt(KNNQueryResult::getId).toArray();
    }
}
