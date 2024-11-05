/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.jni;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.BeforeClass;
import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.RaisingIOExceptionIndexInput;
import org.opensearch.knn.common.RasingIOExceptionIndexOutput;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.nmslib.NmslibHNSWMethod;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QTY;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class JNIServiceTests extends KNNTestCase {
    static final int FP16_MAX = 65504;
    static final int FP16_MIN = -65504;
    static TestUtils.TestData testData;
    static TestUtils.TestData testDataNested;
    private String faissMethod = "HNSW32,Flat";
    private String faissBinaryMethod = "BHNSW32";

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (JNIServiceTests.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of JNIServiceTests Class is null");
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
    public void testCreateIndex_invalid_engineNotSupported() {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                IllegalArgumentException.class,
                () -> TestUtils.createIndex(
                    new int[] {},
                    0,
                    0,
                    directory,
                    "DONT_CARE",
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.LUCENE
                )
            );
        }
    }

    public void testCreateIndex_invalid_engineNull() {
        expectThrows(
            Exception.class,
            () -> TestUtils.createIndex(
                new int[] {},
                0,
                0,
                null,
                "DONT_CARE",
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                null
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_noSpaceType() {
        expectThrows(
            Exception.class,
            () -> TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                null,
                "DONT_CARE",
                Collections.emptyMap(),
                KNNEngine.NMSLIB
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_vectorDocIDMismatch() throws IOException {
        int[] docIds = new int[] { 1, 2, 3 };
        float[][] vectors1 = new float[][] { { 1, 2 }, { 3, 4 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors1, vectors1.length * vectors1[0].length);
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1.tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors1[0].length,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                )
            );

            float[][] vectors2 = new float[][] { { 1, 2 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 8, 9 } };
            long memoryAddress2 = JNICommons.storeVectorData(0, vectors2, vectors2.length * vectors2[0].length);

            String indexFileName2 = "test2.tmp";
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress2,
                    vectors2[0].length,
                    directory,
                    indexFileName2,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                )
            );
        }
    }

    public void testCreateIndex_nmslib_invalid_nullArgument() throws IOException {
        Path tempDirPath = createTempDir();
        String indexFileName = "test.tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] {};
            float[][] vectors = new float[][] {};
            long memoryAddress = JNICommons.storeVectorData(0, vectors, vectors.length);

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    null,
                    memoryAddress,
                    0,
                    directory,
                    indexFileName,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    0,
                    0,
                    directory,
                    indexFileName,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    0,
                    directory,
                    null,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(docIds, memoryAddress, 0, directory, indexFileName, null, KNNEngine.NMSLIB)
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    0,
                    directory,
                    indexFileName,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    null
                )
            );
        }
    }

    public void testCreateIndex_nmslib_invalid_badSpace() throws IOException {

        int[] docIds = new int[] { 1 };
        float[][] vectors = new float[][] { { 2, 3 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, vectors.length * vectors[0].length);
        Path tempDirPath = createTempDir();
        String indexFileName = "test.tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors[0].length,
                    directory,
                    indexFileName,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, "invalid"),
                    KNNEngine.NMSLIB
                )
            );
        }
    }

    public void testCreateIndex_nmslib_invalid_badParameterType() throws IOException {

        int[] docIds = new int[] { 1 };
        float[][] vectors = new float[][] { { 2, 3 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, vectors.length * vectors[0].length);

        Map<String, Object> parametersMap = ImmutableMap.of(
            KNNConstants.HNSW_ALGO_EF_CONSTRUCTION,
            "14",
            KNNConstants.METHOD_PARAMETER_M,
            "12"
        );
        Path tempDirPath = createTempDir();
        String indexFileName = "test.tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors[0].length,
                    directory,
                    indexFileName,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue(), KNNConstants.PARAMETERS, parametersMap),
                    KNNEngine.NMSLIB
                )
            );
        }
    }

    public void testCreateIndex_nmslib_valid() throws IOException {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            for (SpaceType spaceType : NmslibHNSWMethod.SUPPORTED_SPACES) {
                if (SpaceType.UNDEFINED == spaceType) {
                    continue;
                }

                final String indexFileName1 = "test" + UUID.randomUUID() + ".tmp";

                TestUtils.createIndex(
                    testData.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    KNNEngine.NMSLIB
                );
                assertTrue(directory.fileLength(indexFileName1) > 0);

                final String indexFileName2 = "test" + UUID.randomUUID() + ".tmp";

                TestUtils.createIndex(
                    testData.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName2,
                    ImmutableMap.of(
                        KNNConstants.SPACE_TYPE,
                        spaceType.getValue(),
                        KNNConstants.HNSW_ALGO_EF_CONSTRUCTION,
                        14,
                        KNNConstants.METHOD_PARAMETER_M,
                        12
                    ),
                    KNNEngine.NMSLIB
                );
                assertTrue(directory.fileLength(indexFileName2) > 0);
            }
        }
    }

    @SneakyThrows
    public void testCreateIndex_faiss_invalid_noSpaceType() {
        int[] docIds = new int[] {};

        Path tempDirPath = createTempDir();
        String indexFileName = "test.tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod),
                    KNNEngine.FAISS
                )
            );

        }
    }

    public void testCreateIndex_faiss_invalid_vectorDocIDMismatch() throws IOException {

        int[] docIds = new int[] { 1, 2, 3 };
        float[][] vectors1 = new float[][] { { 1, 2 }, { 3, 4 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors1, vectors1.length * vectors1[0].length);
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors1[0].length,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );

            float[][] vectors2 = new float[][] { { 1, 2 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 8, 9 } };
            long memoryAddress2 = JNICommons.storeVectorData(0, vectors2, vectors2.length * vectors2[0].length);
            String indexFileName2 = "test2" + UUID.randomUUID() + ".tmp";
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress2,
                    vectors2[0].length,
                    directory,
                    indexFileName2,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );
        }
    }

    public void testCreateIndex_faiss_invalid_null() throws IOException {
        Path tempDirPath = createTempDir();

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};
        long memoryAddress = JNICommons.storeVectorData(0, vectors, 0);
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    null,
                    memoryAddress,
                    0,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    0,
                    0,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    null,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    null,
                    KNNEngine.FAISS
                )
            );

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    null
                )
            );
        }
    }

    public void testCreateIndex_faiss_invalid_invalidSpace() throws IOException {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] { 1 };
            float[][] vectors = new float[][] { { 2, 3 } };
            long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);
            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";

            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors[0].length,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, "invalid"),
                    KNNEngine.FAISS
                )
            );
        }
    }

    public void testCreateIndex_faiss_invalid_noIndexDescription() throws IOException {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] { 1, 2 };
            float[][] vectors = new float[][] { { 2, 3 }, { 2, 3 } };
            long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);

            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors[0].length,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );
        }
    }

    public void testCreateIndex_faiss_invalid_invalidIndexDescription() throws IOException {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] { 1, 2 };
            float[][] vectors = new float[][] { { 2, 3 }, { 2, 3 } };
            long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);

            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors[0].length,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, "invalid", KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.FAISS
                )
            );
        }
    }

    @SneakyThrows
    public void testCreateIndex_faiss_sqfp16_invalidIndexDescription() {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] { 1, 2 };
            float[][] vectors = new float[][] { { 2, 3 }, { 3, 4 } };
            long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);

            String sqfp16InvalidIndexDescription = "HNSW16,SQfp1655";

            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    memoryAddress,
                    vectors[0].length,
                    directory,
                    indexFileName1,
                    ImmutableMap.of(
                        INDEX_DESCRIPTION_PARAMETER,
                        sqfp16InvalidIndexDescription,
                        KNNConstants.SPACE_TYPE,
                        SpaceType.L2.getValue()
                    ),
                    KNNEngine.FAISS
                )
            );
        }
    }

    @SneakyThrows
    public void testLoadIndex_faiss_sqfp16_valid() {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] { 1, 2 };
            float[][] vectors = new float[][] { { 2, 3 }, { 3, 4 } };
            String sqfp16IndexDescription = "HNSW16,SQfp16";
            long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);
            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            TestUtils.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, sqfp16IndexDescription, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
            }
        }
    }

    @SneakyThrows
    public void testLoadIndex_when_io_exception_was_raised() {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] { 1, 2 };
            float[][] vectors = new float[][] { { 2, 3 }, { 3, 4 } };
            String sqfp16IndexDescription = "HNSW16,SQfp16";
            long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);
            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            TestUtils.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, sqfp16IndexDescription, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final IndexInput raiseIOExceptionIndexInput = new RaisingIOExceptionIndexInput();
            final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(raiseIOExceptionIndexInput);

            try {
                JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                fail("Exception thrown is expected.");
            } catch (Throwable e) {
                assertTrue(e.getMessage().contains("Reading bytes via IndexInput has failed."));
            }
        }
    }

    @SneakyThrows
    public void testQueryIndex_faiss_sqfp16_valid() {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            String sqfp16IndexDescription = "HNSW16,SQfp16";
            int k = 10;
            Map<String, ?> methodParameters = Map.of("ef_search", 12);
            float[][] truncatedVectors = truncateToFp16Range(testData.indexData.vectors);
            long memoryAddress = JNICommons.storeVectorData(
                0,
                truncatedVectors,
                (long) truncatedVectors.length * truncatedVectors[0].length
            );
            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            TestUtils.createIndex(
                testData.indexData.docs,
                memoryAddress,
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, sqfp16IndexDescription, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            for (float[] query : testData.queries) {
                KNNQueryResult[] results = JNIService.queryIndex(pointer, query, k, methodParameters, KNNEngine.FAISS, null, 0, null);
                assertEquals(k, results.length);
            }

            // Filter will result in no ids
            for (float[] query : testData.queries) {
                KNNQueryResult[] results = JNIService.queryIndex(
                    pointer,
                    query,
                    k,
                    methodParameters,
                    KNNEngine.FAISS,
                    new long[] { 0 },
                    0,
                    null
                );
                assertEquals(0, results.length);
            }
        }
    }

    // If the value is outside of the fp16 range, then convert it to the fp16 minimum or maximum value
    private float[][] truncateToFp16Range(final float[][] data) {
        float[][] result = new float[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                float value = data[i][j];
                if (value < FP16_MIN || value > FP16_MAX) {
                    // If value is outside of the range, set it to the maximum or minimum value
                    result[i][j] = value < 0 ? FP16_MIN : FP16_MAX;
                } else {
                    result[i][j] = value;
                }
            }
        }
        return result;
    }

    @SneakyThrows
    public void testTrain_whenConfigurationIsIVFSQFP16_thenSucceed() {
        long trainPointer = transferVectors(10);
        int ivfNlistParam = 16;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, ivfNlistParam)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(128)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        Map<String, Object> parameters = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
    }

    public void testCreateIndex_faiss_invalid_invalidParameterType() throws IOException {
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int[] docIds = new int[] {};
            float[][] vectors = new float[][] {};

            String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
            expectThrows(
                Exception.class,
                () -> TestUtils.createIndex(
                    docIds,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    ImmutableMap.of(
                        INDEX_DESCRIPTION_PARAMETER,
                        "IVF13",
                        KNNConstants.SPACE_TYPE,
                        SpaceType.L2.getValue(),
                        KNNConstants.PARAMETERS,
                        ImmutableMap.of(KNNConstants.METHOD_PARAMETER_NPROBES, "14")
                    ),
                    KNNEngine.FAISS
                )
            );
        }
    }

    public void testCreateIndex_faiss_valid() throws IOException {

        List<String> methods = ImmutableList.of(faissMethod);
        List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            for (String method : methods) {
                for (SpaceType spaceType : spaces) {
                    String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                    TestUtils.createIndex(
                        testData.indexData.docs,
                        testData.loadDataToMemoryAddress(),
                        testData.indexData.getDimension(),
                        directory,
                        indexFileName1,
                        ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                        KNNEngine.FAISS
                    );
                    assertTrue(directory.fileLength(indexFileName1) > 0);
                }
            }
        }
    }

    @SneakyThrows
    public void testCreateIndex_binary_faiss_valid() {
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            long memoryAddr = testData.loadBinaryDataToMemoryAddress();
            TestUtils.createIndex(
                testData.indexData.docs,
                memoryAddr,
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
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
            assertTrue(directory.fileLength(indexFileName1) > 0);
        }
    }

    public void testLoadIndex_invalidEngine() {
        expectThrows(IllegalArgumentException.class, () -> JNIService.loadIndex(null, Collections.emptyMap(), KNNEngine.LUCENE));
    }

    public void testLoadIndex_nmslib_invalid_badSpaceType() {
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(null, ImmutableMap.of(KNNConstants.SPACE_TYPE, "invalid"), KNNEngine.NMSLIB)
        );
    }

    public void testLoadIndex_nmslib_invalid_noSpaceType() {
        expectThrows(Exception.class, () -> JNIService.loadIndex(null, Collections.emptyMap(), KNNEngine.NMSLIB));
    }

    public void testLoadIndex_nmslib_invalid_fileDoesNotExist() {
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(null, ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()), KNNEngine.NMSLIB)
        );
    }

    public void testLoadIndex_nmslib_invalid_badFile() throws IOException {
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(null, ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()), KNNEngine.NMSLIB)
        );
    }

    public void testLoadIndex_nmslib_valid() throws IOException {

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long pointer = JNIService.loadIndex(
                    indexInputWithBuffer,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                );
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
            }
        }
    }

    public void testLoadIndex_nmslib_raise_io_exception() throws IOException {

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final IndexInput raiseIOExceptionIndexInput = new RaisingIOExceptionIndexInput();

            final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(raiseIOExceptionIndexInput);
            try {
                JNIService.loadIndex(
                    indexInputWithBuffer,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                );
                fail("Exception expected");
            } catch (Throwable e) {
                assertTrue(e.getMessage().contains("Reading bytes via IndexInput has failed."));
            }
        }
    }

    public void testLoadIndex_nmslib_valid_with_stream() throws IOException {
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long pointer = JNIService.loadIndex(
                    indexInputWithBuffer,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                );
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
            }
        }
    }

    public void testWriteIndex_nmslib_when_io_exception_occured() {
        try {
            final IndexOutput indexOutput = new RasingIOExceptionIndexOutput();
            final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOutput);
            JNIService.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                indexOutputWithBuffer,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            );
            fail("Exception thrown is expected.");
        } catch (Throwable e) {
            assertTrue(e.getMessage().contains("Writing bytes via IndexOutput has failed."));
        }
    }

    public void testLoadIndex_faiss_valid() throws IOException {
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
            }
        }
    }

    public void testQueryIndex_invalidEngine() {
        expectThrows(
            IllegalArgumentException.class,
            () -> JNIService.queryIndex(0L, new float[] {}, 0, null, KNNEngine.LUCENE, null, 0, null)
        );
    }

    public void testQueryIndex_nmslib_invalid_badPointer() {

        expectThrows(Exception.class, () -> JNIService.queryIndex(0L, new float[] {}, 0, null, KNNEngine.NMSLIB, null, 0, null));
    }

    public void testQueryIndex_nmslib_invalid_nullQueryVector() throws IOException {

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(
                    indexInputWithBuffer,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                );
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, null, KNNEngine.NMSLIB, null, 0, null));
        }
    }

    public void testQueryIndex_nmslib_valid() throws IOException {

        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            int k = 50;
            for (SpaceType spaceType : NmslibHNSWMethod.SUPPORTED_SPACES) {
                if (SpaceType.UNDEFINED == spaceType) {
                    continue;
                }

                String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";

                TestUtils.createIndex(
                    testData.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    KNNEngine.NMSLIB
                );
                assertTrue(directory.fileLength(indexFileName1) > 0);

                final long pointer;
                try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                    final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                    pointer = JNIService.loadIndex(
                        indexInputWithBuffer,
                        ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                        KNNEngine.NMSLIB
                    );
                    assertNotEquals(0, pointer);
                } catch (Throwable e) {
                    fail(e.getMessage());
                    throw e;
                }

                for (float[] query : testData.queries) {
                    KNNQueryResult[] results = JNIService.queryIndex(pointer, query, k, null, KNNEngine.NMSLIB, null, 0, null);
                    assertEquals(k, results.length);
                }
            }
        }
    }

    public void testQueryIndex_faiss_invalid_badPointer() {

        expectThrows(Exception.class, () -> JNIService.queryIndex(0L, new float[] {}, 0, null, KNNEngine.FAISS, null, 0, null));
    }

    public void testQueryIndex_faiss_invalid_nullQueryVector() throws IOException {

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, null, KNNEngine.FAISS, null, 0, null));
        }
    }

    public void testQueryIndex_faiss_streaming_invalid_nullQueryVector() throws IOException {
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, null, KNNEngine.FAISS, null, 0, null));
        }
    }

    public void testQueryIndex_faiss_valid() throws IOException {

        int k = 10;
        int efSearch = 100;

        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            List<String> methods = ImmutableList.of(faissMethod);
            List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
            for (String method : methods) {
                for (SpaceType spaceType : spaces) {
                    String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                    TestUtils.createIndex(
                        testData.indexData.docs,
                        testData.loadDataToMemoryAddress(),
                        testData.indexData.getDimension(),
                        directory,
                        indexFileName1,
                        ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                        KNNEngine.FAISS
                    );
                    assertTrue(directory.fileLength(indexFileName1) > 0);

                    final long pointer;
                    try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                        pointer = JNIService.loadIndex(
                            indexInputWithBuffer,
                            ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                            KNNEngine.FAISS
                        );
                        assertNotEquals(0, pointer);
                    } catch (Throwable e) {
                        fail(e.getMessage());
                        throw e;
                    }

                    for (float[] query : testData.queries) {
                        KNNQueryResult[] results = JNIService.queryIndex(
                            pointer,
                            query,
                            k,
                            Map.of("ef_search", efSearch),
                            KNNEngine.FAISS,
                            null,
                            0,
                            null
                        );
                        assertEquals(k, results.length);
                    }

                    // Filter will result in no ids
                    for (float[] query : testData.queries) {
                        KNNQueryResult[] results = JNIService.queryIndex(
                            pointer,
                            query,
                            k,
                            Map.of("ef_search", efSearch),
                            KNNEngine.FAISS,
                            new long[] { 0 },
                            0,
                            null
                        );
                        assertEquals(0, results.length);
                    }
                }
            }
        }
    }

    public void testQueryIndex_faiss_streaming_valid() throws IOException {
        int k = 10;
        int efSearch = 100;

        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            List<String> methods = ImmutableList.of(faissMethod);
            List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
            for (String method : methods) {
                for (SpaceType spaceType : spaces) {
                    String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                    TestUtils.createIndex(
                        testData.indexData.docs,
                        testData.loadDataToMemoryAddress(),
                        testData.indexData.getDimension(),
                        directory,
                        indexFileName1,
                        ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                        KNNEngine.FAISS
                    );
                    assertTrue(directory.fileLength(indexFileName1) > 0);

                    try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.READONCE)) {
                        long pointer = JNIService.loadIndex(
                            new IndexInputWithBuffer(indexInput),
                            ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                            KNNEngine.FAISS
                        );
                        assertNotEquals(0, pointer);

                        for (float[] query : testData.queries) {
                            KNNQueryResult[] results = JNIService.queryIndex(
                                pointer,
                                query,
                                k,
                                Map.of("ef_search", efSearch),
                                KNNEngine.FAISS,
                                null,
                                0,
                                null
                            );
                            assertEquals(k, results.length);
                        }

                        // Filter will result in no ids
                        for (float[] query : testData.queries) {
                            KNNQueryResult[] results = JNIService.queryIndex(
                                pointer,
                                query,
                                k,
                                Map.of("ef_search", efSearch),
                                KNNEngine.FAISS,
                                new long[] { 0 },
                                0,
                                null
                            );
                            assertEquals(0, results.length);
                        }  // End for
                    }  // End try
                }  // End for
            }  // End for
        }
    }

    public void testQueryIndex_faiss_parentIds() throws IOException {

        int k = 100;
        int efSearch = 100;

        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            List<String> methods = ImmutableList.of(faissMethod);
            List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
            int[] parentIds = toParentIdArray(testDataNested.indexData.docs);
            Map<Integer, Integer> idToParentIdMap = toIdToParentIdMap(testDataNested.indexData.docs);
            for (String method : methods) {
                for (SpaceType spaceType : spaces) {
                    String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                    TestUtils.createIndex(
                        testDataNested.indexData.docs,
                        testData.loadDataToMemoryAddress(),
                        testDataNested.indexData.getDimension(),
                        directory,
                        indexFileName1,
                        ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                        KNNEngine.FAISS
                    );
                    assertTrue(directory.fileLength(indexFileName1) > 0);

                    final long pointer;
                    try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                        pointer = JNIService.loadIndex(
                            indexInputWithBuffer,
                            ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                            KNNEngine.FAISS
                        );
                        assertNotEquals(0, pointer);
                    } catch (Throwable e) {
                        fail(e.getMessage());
                        throw e;
                    }

                    for (float[] query : testDataNested.queries) {
                        KNNQueryResult[] results = JNIService.queryIndex(
                            pointer,
                            query,
                            k,
                            Map.of("ef_search", efSearch),
                            KNNEngine.FAISS,
                            null,
                            0,
                            parentIds
                        );
                        // Verify there is no more than one result from same parent
                        Set<Integer> parentIdSet = toParentIdSet(results, idToParentIdMap);
                        assertEquals(results.length, parentIdSet.size());
                    }
                }
            }
        }
    }

    public void testQueryIndex_faiss_streaming_parentIds() throws IOException {

        int k = 100;
        int efSearch = 100;

        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            List<String> methods = ImmutableList.of(faissMethod);
            List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
            int[] parentIds = toParentIdArray(testDataNested.indexData.docs);
            Map<Integer, Integer> idToParentIdMap = toIdToParentIdMap(testDataNested.indexData.docs);
            for (String method : methods) {
                for (SpaceType spaceType : spaces) {
                    String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                    TestUtils.createIndex(
                        testDataNested.indexData.docs,
                        testData.loadDataToMemoryAddress(),
                        testDataNested.indexData.getDimension(),
                        directory,
                        indexFileName1,
                        ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                        KNNEngine.FAISS
                    );
                    assertTrue(directory.fileLength(indexFileName1) > 0);

                    try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.READONCE)) {
                        long pointer = JNIService.loadIndex(
                            new IndexInputWithBuffer(indexInput),
                            ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                            KNNEngine.FAISS
                        );
                        assertNotEquals(0, pointer);

                        for (float[] query : testDataNested.queries) {
                            KNNQueryResult[] results = JNIService.queryIndex(
                                pointer,
                                query,
                                k,
                                Map.of("ef_search", efSearch),
                                KNNEngine.FAISS,
                                null,
                                0,
                                parentIds
                            );
                            // Verify there is no more than one result from same parent
                            Set<Integer> parentIdSet = toParentIdSet(results, idToParentIdMap);
                            assertEquals(results.length, parentIdSet.size());
                        }  // End for
                    }  // End try
                }  // End for
            }  // End for
        }
    }

    @SneakyThrows
    public void testQueryBinaryIndex_faiss_valid() {
        int k = 10;
        List<String> methods = ImmutableList.of(faissBinaryMethod);
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            for (String method : methods) {
                String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                long memoryAddr = testData.loadBinaryDataToMemoryAddress();
                TestUtils.createIndex(
                    testData.indexData.docs,
                    memoryAddr,
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    ImmutableMap.of(
                        INDEX_DESCRIPTION_PARAMETER,
                        method,
                        KNNConstants.SPACE_TYPE,
                        SpaceType.HAMMING.getValue(),
                        KNNConstants.VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BINARY.getValue()
                    ),
                    KNNEngine.FAISS
                );
                assertTrue(directory.fileLength(indexFileName1) > 0);

                final long pointer;
                try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                    final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                    pointer = JNIService.loadIndex(
                        indexInputWithBuffer,
                        ImmutableMap.of(
                            INDEX_DESCRIPTION_PARAMETER,
                            method,
                            KNNConstants.VECTOR_DATA_TYPE_FIELD,
                            VectorDataType.BINARY.getValue()
                        ),
                        KNNEngine.FAISS
                    );
                    assertNotEquals(0, pointer);
                } catch (Throwable e) {
                    fail(e.getMessage());
                    throw e;
                }

                for (byte[] query : testData.binaryQueries) {
                    KNNQueryResult[] results = JNIService.queryBinaryIndex(pointer, query, k, null, KNNEngine.FAISS, null, 0, null);
                    assertEquals(k, results.length);
                }
            }
        }
    }

    @SneakyThrows
    public void testQueryBinaryIndex_faiss_streaming_valid() {
        int k = 10;
        List<String> methods = ImmutableList.of(faissBinaryMethod);
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            for (String method : methods) {
                String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
                long memoryAddr = testData.loadBinaryDataToMemoryAddress();
                TestUtils.createIndex(
                    testData.indexData.docs,
                    memoryAddr,
                    testData.indexData.getDimension(),
                    directory,
                    indexFileName1,
                    ImmutableMap.of(
                        INDEX_DESCRIPTION_PARAMETER,
                        method,
                        KNNConstants.SPACE_TYPE,
                        SpaceType.HAMMING.getValue(),
                        KNNConstants.VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BINARY.getValue()
                    ),
                    KNNEngine.FAISS
                );
                assertTrue(directory.fileLength(indexFileName1) > 0);

                try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.READONCE)) {
                    long pointer = JNIService.loadIndex(
                        new IndexInputWithBuffer(indexInput),
                        ImmutableMap.of(
                            INDEX_DESCRIPTION_PARAMETER,
                            method,
                            KNNConstants.VECTOR_DATA_TYPE_FIELD,
                            VectorDataType.BINARY.getValue()
                        ),
                        KNNEngine.FAISS
                    );
                    assertNotEquals(0, pointer);

                    for (byte[] query : testData.binaryQueries) {
                        KNNQueryResult[] results = JNIService.queryBinaryIndex(pointer, query, k, null, KNNEngine.FAISS, null, 0, null);
                        assertEquals(k, results.length);
                    }  // End for
                }  // End try
            }  // End for
        }  // End try
    }

    private Set<Integer> toParentIdSet(KNNQueryResult[] results, Map<Integer, Integer> idToParentIdMap) {
        return Arrays.stream(results).map(result -> idToParentIdMap.get(result.getId())).collect(Collectors.toSet());
    }

    private int[] toParentIdArray(int[] ids) {
        int length = ids.length;
        int[] sortedIds = Arrays.copyOf(ids, length);
        Arrays.sort(sortedIds);

        List<Integer> parentIds = new ArrayList<>();
        int largestId = sortedIds[length - 1];
        parentIds.add(largestId + 1);
        for (int i = length - 2; i > -1; i--) {
            if (sortedIds[i] != sortedIds[i + 1] - 1) {
                parentIds.add(sortedIds[i] + 1);
            }
        }

        Collections.shuffle(parentIds);
        return parentIds.stream().mapToInt(Integer::intValue).toArray();
    }

    private Map<Integer, Integer> toIdToParentIdMap(int[] ids) {
        int length = ids.length;
        int[] sortedIds = Arrays.copyOf(ids, length);
        Arrays.sort(sortedIds);

        Map<Integer, Integer> idToParentIdMap = new HashMap<>();
        int largestId = sortedIds[length - 1];
        int parentId = largestId + 1;
        idToParentIdMap.put(largestId, parentId);
        for (int i = length - 2; i > -1; i--) {
            if (sortedIds[i] != sortedIds[i + 1] - 1) {
                parentId = sortedIds[i] + 1;
            }
            idToParentIdMap.put(sortedIds[i], parentId);
        }
        return idToParentIdMap;
    }

    public void testFree_invalidEngine() {
        expectThrows(IllegalArgumentException.class, () -> JNIService.free(0L, KNNEngine.LUCENE));
    }

    public void testFree_nmslib_valid() throws IOException {

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(
                    indexInputWithBuffer,
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                    KNNEngine.NMSLIB
                );
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            JNIService.free(pointer, KNNEngine.NMSLIB);
        }
    }

    public void testFree_faiss_valid() throws IOException {

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            TestUtils.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            JNIService.free(pointer, KNNEngine.FAISS);
        }
    }

    public void testTransferVectors() {
        long trainPointer1 = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer1);

        long trainPointer2;
        for (int i = 0; i < 10; i++) {
            trainPointer2 = JNIService.transferVectors(trainPointer1, testData.indexData.vectors);
            assertEquals(trainPointer1, trainPointer2);
        }

        JNICommons.freeVectorData(trainPointer1);
    }

    public void testTrain_whenConfigurationIsIVFFlat_thenSucceed() throws IOException {
        long trainPointer = transferVectors(10);
        int ivfNlistParam = 16;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, ivfNlistParam)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(testData.indexData.getDimension())
            .versionCreated(Version.CURRENT)
            .build();
        Map<String, Object> parameters = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
    }

    public void testTrain_whenConfigurationIsIVFPQ_thenSucceed() throws IOException {
        long trainPointer = transferVectors(10);
        int ivfNlistParam = 16;
        int pqMParam = 4;
        int pqCodeSizeParam = 4;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, ivfNlistParam)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, pqMParam)
            .field(ENCODER_PARAMETER_PQ_CODE_SIZE, pqCodeSizeParam)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(128)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        Map<String, Object> parameters = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
    }

    public void testTrain_whenConfigurationIsHNSWPQ_thenSucceed() throws IOException {
        long trainPointer = transferVectors(10);
        int pqMParam = 4;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT.getValue())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, pqMParam)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(testData.indexData.getDimension())
            .versionCreated(Version.CURRENT)
            .build();
        Map<String, Object> parameters = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
    }

    private long transferVectors(int numDuplicates) {
        long trainPointer1 = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer1);

        long trainPointer2;
        for (int i = 0; i < numDuplicates; i++) {
            trainPointer2 = JNIService.transferVectors(trainPointer1, testData.indexData.vectors);
            assertEquals(trainPointer1, trainPointer2);
        }

        return trainPointer1;
    }

    public void createIndexFromTemplate() throws IOException {

        long trainPointer1 = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer1);

        long trainPointer2;
        for (int i = 0; i < 10; i++) {
            trainPointer2 = JNIService.transferVectors(trainPointer1, testData.indexData.vectors);
            assertEquals(trainPointer1, trainPointer2);
        }

        SpaceType spaceType = SpaceType.L2;
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(128)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            spaceType,
            new MethodComponentContext(
                METHOD_IVF,
                ImmutableMap.of(
                    METHOD_PARAMETER_NLIST,
                    16,
                    METHOD_ENCODER_PARAMETER,
                    new MethodComponentContext(ENCODER_PQ, ImmutableMap.of(ENCODER_PARAMETER_PQ_M, 16, ENCODER_PARAMETER_PQ_CODE_SIZE, 8))
                )
            )
        );

        String description = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters()
            .get(INDEX_DESCRIPTION_PARAMETER)
            .toString();
        assertEquals("IVF16,PQ16x8", description);

        Map<String, Object> parameters = ImmutableMap.of(
            INDEX_DESCRIPTION_PARAMETER,
            description,
            KNNConstants.SPACE_TYPE,
            spaceType.getValue()
        );

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer1, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer1);

        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            try (IndexOutput indexOutput = directory.createOutput(indexFileName1, IOContext.DEFAULT)) {
                final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOutput);
                JNIService.createIndexFromTemplate(
                    testData.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    indexOutputWithBuffer,
                    faissIndex,
                    ImmutableMap.of(INDEX_THREAD_QTY, 1),
                    KNNEngine.FAISS
                );
            }
            assertTrue(directory.fileLength(indexFileName1) > 0);

            final long pointer;
            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                pointer = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, pointer);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }
        }
    }

    @SneakyThrows
    public void testCreateIndex_whenIOExceptionOccured() {
        // Plain index
        Map<String, Object> parameters = new HashMap<>(
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue())
        );

        long trainPointer = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(128)
            .vectorDataType(VectorDataType.FLOAT)
            .build();

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);

        final IndexOutput indexOutput = new RasingIOExceptionIndexOutput();
        final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOutput);
        try {
            JNIService.createIndexFromTemplate(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                indexOutputWithBuffer,
                faissIndex,
                ImmutableMap.of(INDEX_THREAD_QTY, 1),
                KNNEngine.FAISS
            );
            fail("Exception thrown was expected");
        } catch (Throwable t) {
            System.out.println("!!!!!!!!!!!!!!!!!!!!! " + t.getMessage());
        }
    }

    @SneakyThrows
    public void testIndexLoad_whenStateIsShared_thenSucceed() {
        // Creates a single IVFPQ-l2 index. Then, we will configure a set of indices in memory in different ways to
        // ensure that everything is loaded properly and the results are consistent.
        int k = 10;
        int ivfNlist = 16;
        int pqM = 16;
        int pqCodeSize = 4;
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            String indexIVFPQPath = createFaissIVFPQIndex(directory, ivfNlist, pqM, pqCodeSize, SpaceType.L2);

            final long indexIVFPQIndexTest1;
            try (IndexInput indexInput = directory.openInput(indexIVFPQPath, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                indexIVFPQIndexTest1 = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, indexIVFPQIndexTest1);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }
            final long indexIVFPQIndexTest2;
            try (IndexInput indexInput = directory.openInput(indexIVFPQPath, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                indexIVFPQIndexTest2 = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, indexIVFPQIndexTest2);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            long sharedStateAddress = JNIService.initSharedIndexState(indexIVFPQIndexTest1, KNNEngine.FAISS);
            JNIService.setSharedIndexState(indexIVFPQIndexTest1, sharedStateAddress, KNNEngine.FAISS);
            JNIService.setSharedIndexState(indexIVFPQIndexTest2, sharedStateAddress, KNNEngine.FAISS);

            assertQueryResultsMatch(testData.queries, k, List.of(indexIVFPQIndexTest1, indexIVFPQIndexTest2));

            // Free the first test index 1. This will ensure that the shared state persists after index that initialized
            // shared state is gone.
            JNIService.free(indexIVFPQIndexTest1, KNNEngine.FAISS);

            final long indexIVFPQIndexTest3;
            try (IndexInput indexInput = directory.openInput(indexIVFPQPath, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                indexIVFPQIndexTest3 = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertNotEquals(0, indexIVFPQIndexTest3);
            } catch (Throwable e) {
                fail(e.getMessage());
                throw e;
            }

            JNIService.setSharedIndexState(indexIVFPQIndexTest3, sharedStateAddress, KNNEngine.FAISS);

            assertQueryResultsMatch(testData.queries, k, List.of(indexIVFPQIndexTest2, indexIVFPQIndexTest3));

            // Ensure everything gets freed
            JNIService.free(indexIVFPQIndexTest2, KNNEngine.FAISS);
            JNIService.free(indexIVFPQIndexTest3, KNNEngine.FAISS);
            JNIService.freeSharedIndexState(sharedStateAddress, KNNEngine.FAISS);
        }
    }

    @SneakyThrows
    public void testIsIndexIVFPQL2() {
        long dummyAddress = 0;
        assertFalse(JNIService.isSharedIndexStateRequired(dummyAddress, KNNEngine.NMSLIB));

        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            String faissIVFPQL2Index = createFaissIVFPQIndex(directory, 16, 16, 4, SpaceType.L2);
            try (IndexInput indexInput = directory.openInput(faissIVFPQL2Index, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long faissIVFPQL2Address = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertTrue(JNIService.isSharedIndexStateRequired(faissIVFPQL2Address, KNNEngine.FAISS));
                JNIService.free(faissIVFPQL2Address, KNNEngine.FAISS);
            }

            String faissIVFPQIPIndex = createFaissIVFPQIndex(directory, 16, 16, 4, SpaceType.INNER_PRODUCT);
            try (IndexInput indexInput = directory.openInput(faissIVFPQIPIndex, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long faissIVFPQIPAddress = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertFalse(JNIService.isSharedIndexStateRequired(faissIVFPQIPAddress, KNNEngine.FAISS));
                JNIService.free(faissIVFPQIPAddress, KNNEngine.FAISS);
            }

            String faissHNSWIndex = createFaissHNSWIndex(directory, SpaceType.L2);
            try (IndexInput indexInput = directory.openInput(faissHNSWIndex, IOContext.LOAD)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                long faissHNSWAddress = JNIService.loadIndex(indexInputWithBuffer, Collections.emptyMap(), KNNEngine.FAISS);
                assertFalse(JNIService.isSharedIndexStateRequired(faissHNSWAddress, KNNEngine.FAISS));
                JNIService.free(faissHNSWAddress, KNNEngine.FAISS);
            }
        }
    }

    @SneakyThrows
    public void testFunctionsUnsupportedForEngine_whenEngineUnsupported_thenThrowIllegalArgumentException() {
        int dummyAddress = 0;
        expectThrows(IllegalArgumentException.class, () -> JNIService.initSharedIndexState(dummyAddress, KNNEngine.NMSLIB));
        expectThrows(IllegalArgumentException.class, () -> JNIService.setSharedIndexState(dummyAddress, dummyAddress, KNNEngine.NMSLIB));
        expectThrows(IllegalArgumentException.class, () -> JNIService.freeSharedIndexState(dummyAddress, KNNEngine.NMSLIB));
    }

    private void assertQueryResultsMatch(float[][] testQueries, int k, List<Long> indexAddresses) {
        // Checks that the set of queries is consistent amongst all indices in the list
        for (float[] query : testQueries) {
            KNNQueryResult[][] allResults = new KNNQueryResult[indexAddresses.size()][];
            for (int i = 0; i < indexAddresses.size(); i++) {
                allResults[i] = JNIService.queryIndex(indexAddresses.get(i), query, k, null, KNNEngine.FAISS, null, 0, null);
                assertEquals(k, allResults[i].length);
            }

            for (int i = 1; i < indexAddresses.size(); i++) {
                for (int j = 0; j < k; j++) {
                    assertEquals(allResults[0][j].getId(), allResults[i][j].getId());
                    assertEquals(allResults[0][j].getScore(), allResults[i][j].getScore(), 0.00001);
                }
            }
        }
    }

    private String createFaissIVFPQIndex(Directory directory, int ivfNlist, int pqM, int pqCodeSize, SpaceType spaceType)
        throws IOException {
        long trainPointer = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(128)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            spaceType,
            new MethodComponentContext(
                METHOD_IVF,
                ImmutableMap.of(
                    METHOD_PARAMETER_NLIST,
                    ivfNlist,
                    METHOD_ENCODER_PARAMETER,
                    new MethodComponentContext(
                        ENCODER_PQ,
                        ImmutableMap.of(ENCODER_PARAMETER_PQ_M, pqM, ENCODER_PARAMETER_PQ_CODE_SIZE, pqCodeSize)
                    )
                )
            )
        );

        String description = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters()
            .get(INDEX_DESCRIPTION_PARAMETER)
            .toString();
        Map<String, Object> parameters = ImmutableMap.of(
            INDEX_DESCRIPTION_PARAMETER,
            description,
            KNNConstants.SPACE_TYPE,
            spaceType.getValue()
        );

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (IndexOutput indexOutput = directory.createOutput(indexFileName1, IOContext.DEFAULT)) {
            final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOutput);
            JNIService.createIndexFromTemplate(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                indexOutputWithBuffer,
                faissIndex,
                ImmutableMap.of(INDEX_THREAD_QTY, 1),
                KNNEngine.FAISS
            );
        }
        assertTrue(directory.fileLength(indexFileName1) > 0);

        return indexFileName1;
    }

    private String createFaissHNSWIndex(Directory directory, SpaceType spaceType) throws IOException {
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        TestUtils.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            directory,
            indexFileName1,
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, spaceType.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(directory.fileLength(indexFileName1) > 0);
        return indexFileName1;
    }
}
