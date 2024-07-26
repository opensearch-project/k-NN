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
import org.junit.BeforeClass;
import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

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

    public void testCreateIndex_invalid_engineNotSupported() {
        expectThrows(
            IllegalArgumentException.class,
            () -> JNIService.createIndex(
                new int[] {},
                0,
                0,
                "test",
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.LUCENE
            )
        );
    }

    public void testCreateIndex_invalid_engineNull() {
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                new int[] {},
                0,
                0,
                "test",
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                null
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_noSpaceType() {
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                "something",
                Collections.emptyMap(),
                KNNEngine.NMSLIB
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_vectorDocIDMismatch() throws IOException {

        int[] docIds = new int[] { 1, 2, 3 };
        float[][] vectors1 = new float[][] { { 1, 2 }, { 3, 4 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors1, vectors1.length * vectors1[0].length);
        Path tmpFile1 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors1[0].length,
                tmpFile1.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            )
        );

        float[][] vectors2 = new float[][] { { 1, 2 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 8, 9 } };
        long memoryAddress2 = JNICommons.storeVectorData(0, vectors2, vectors2.length * vectors2[0].length);

        Path tmpFile2 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress2,
                vectors2[0].length,
                tmpFile2.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_nullArgument() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};
        long memoryAddress = JNICommons.storeVectorData(0, vectors, vectors.length);
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                null,
                memoryAddress,
                0,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                0,
                0,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                0,
                null,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(docIds, memoryAddress, 0, tmpFile.toAbsolutePath().toString(), null, KNNEngine.NMSLIB)
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                0,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                null
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_badSpace() throws IOException {

        int[] docIds = new int[] { 1 };
        float[][] vectors = new float[][] { { 2, 3 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, vectors.length * vectors[0].length);
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, "invalid"),
                KNNEngine.NMSLIB
            )
        );
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
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue(), KNNConstants.PARAMETERS, parametersMap),
                KNNEngine.NMSLIB
            )
        );
    }

    public void testCreateIndex_nmslib_valid() throws IOException {

        for (SpaceType spaceType : KNNEngine.NMSLIB.getMethod(KNNConstants.METHOD_HNSW).getSpaces()) {
            if (SpaceType.UNDEFINED == spaceType) {
                continue;
            }

            Path tmpFile = createTempFile();

            JNIService.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(tmpFile.toFile().length() > 0);

            tmpFile = createTempFile();

            JNIService.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
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
            assertTrue(tmpFile.toFile().length() > 0);
        }
    }

    public void testCreateIndex_faiss_invalid_noSpaceType() {
        int[] docIds = new int[] {};

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                "something",
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod),
                KNNEngine.FAISS
            )
        );
    }

    public void testCreateIndex_faiss_invalid_vectorDocIDMismatch() throws IOException {

        int[] docIds = new int[] { 1, 2, 3 };
        float[][] vectors1 = new float[][] { { 1, 2 }, { 3, 4 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors1, vectors1.length * vectors1[0].length);
        Path tmpFile1 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors1[0].length,
                tmpFile1.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );

        float[][] vectors2 = new float[][] { { 1, 2 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 8, 9 } };
        long memoryAddress2 = JNICommons.storeVectorData(0, vectors2, vectors2.length * vectors2[0].length);
        Path tmpFile2 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors2[0].length,
                tmpFile2.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );
    }

    public void testCreateIndex_faiss_invalid_null() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};
        long memoryAddress = JNICommons.storeVectorData(0, vectors, 0);

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                null,
                memoryAddress,
                0,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                0,
                0,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                null,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
                null,
                KNNEngine.FAISS
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                null
            )
        );
    }

    public void testCreateIndex_faiss_invalid_invalidSpace() throws IOException {

        int[] docIds = new int[] { 1 };
        float[][] vectors = new float[][] { { 2, 3 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, "invalid"),
                KNNEngine.FAISS
            )
        );
    }

    public void testCreateIndex_faiss_invalid_noIndexDescription() throws IOException {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 2, 3 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );
    }

    public void testCreateIndex_faiss_invalid_invalidIndexDescription() throws IOException {
        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 2, 3 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, "invalid", KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.FAISS
            )
        );
    }

    @SneakyThrows
    public void testCreateIndex_faiss_sqfp16_invalidIndexDescription() {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 3, 4 } };
        long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);

        String sqfp16InvalidIndexDescription = "HNSW16,SQfp1655";

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                memoryAddress,
                vectors[0].length,
                tmpFile.toAbsolutePath().toString(),
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

    @SneakyThrows
    public void testLoadIndex_faiss_sqfp16_valid() {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 3, 4 } };
        String sqfp16IndexDescription = "HNSW16,SQfp16";
        long memoryAddress = JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);
        Path tmpFile = createTempFile();
        JNIService.createIndex(
            docIds,
            memoryAddress,
            vectors[0].length,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, sqfp16IndexDescription, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, pointer);
    }

    @SneakyThrows
    public void testQueryIndex_faiss_sqfp16_valid() {

        String sqfp16IndexDescription = "HNSW16,SQfp16";
        int k = 10;
        Map<String, ?> methodParameters = Map.of("ef_search", 12);
        float[][] truncatedVectors = truncateToFp16Range(testData.indexData.vectors);
        long memoryAddress = JNICommons.storeVectorData(0, truncatedVectors, (long) truncatedVectors.length * truncatedVectors[0].length);
        Path tmpFile = createTempFile();
        JNIService.createIndex(
            testData.indexData.docs,
            memoryAddress,
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, sqfp16IndexDescription, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, pointer);

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
        Map<String, Object> parameters = KNNEngine.FAISS.getMethodAsMap(knnMethodContext);

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
    }

    public void testCreateIndex_faiss_invalid_invalidParameterType() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
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

    public void testCreateIndex_faiss_valid() throws IOException {

        List<String> methods = ImmutableList.of(faissMethod);
        List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        for (String method : methods) {
            for (SpaceType spaceType : spaces) {
                Path tmpFile1 = createTempFile();
                JNIService.createIndex(
                    testData.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    tmpFile1.toAbsolutePath().toString(),
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    KNNEngine.FAISS
                );
                assertTrue(tmpFile1.toFile().length() > 0);
            }
        }
    }

    @SneakyThrows
    public void testCreateIndex_binary_faiss_valid() {
        Path tmpFile1 = createTempFile();
        long memoryAddr = testData.loadBinaryDataToMemoryAddress();
        JNIService.createIndex(
            testData.indexData.docs,
            memoryAddr,
            testData.indexData.getDimension(),
            tmpFile1.toAbsolutePath().toString(),
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
        assertTrue(tmpFile1.toFile().length() > 0);
    }

    public void testLoadIndex_invalidEngine() {
        expectThrows(IllegalArgumentException.class, () -> JNIService.loadIndex("test", Collections.emptyMap(), KNNEngine.LUCENE));
    }

    public void testLoadIndex_nmslib_invalid_badSpaceType() {
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex("test", ImmutableMap.of(KNNConstants.SPACE_TYPE, "invalid"), KNNEngine.NMSLIB)
        );
    }

    public void testLoadIndex_nmslib_invalid_noSpaceType() {
        expectThrows(Exception.class, () -> JNIService.loadIndex("test", Collections.emptyMap(), KNNEngine.NMSLIB));
    }

    public void testLoadIndex_nmslib_invalid_fileDoesNotExist() {
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex("invalid", ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()), KNNEngine.NMSLIB)
        );
    }

    public void testLoadIndex_nmslib_invalid_badFile() throws IOException {
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB
            )
        );
    }

    public void testLoadIndex_nmslib_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB
        );
        assertNotEquals(0, pointer);
    }

    public void testLoadIndex_faiss_invalid_fileDoesNotExist() {
        expectThrows(Exception.class, () -> JNIService.loadIndex("invalid", Collections.emptyMap(), KNNEngine.FAISS));
    }

    public void testLoadIndex_faiss_invalid_badFile() throws IOException {

        Path tmpFile = createTempFile();

        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS)
        );
    }

    public void testLoadIndex_faiss_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, pointer);
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

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB
        );
        assertNotEquals(0, pointer);

        expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, null, KNNEngine.NMSLIB, null, 0, null));
    }

    public void testQueryIndex_nmslib_valid() throws IOException {

        int k = 50;
        for (SpaceType spaceType : KNNEngine.NMSLIB.getMethod(KNNConstants.METHOD_HNSW).getSpaces()) {
            if (SpaceType.UNDEFINED == spaceType) {
                continue;
            }

            Path tmpFile = createTempFile();

            JNIService.createIndex(
                testData.indexData.docs,
                testData.loadDataToMemoryAddress(),
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                KNNEngine.NMSLIB
            );
            assertTrue(tmpFile.toFile().length() > 0);

            long pointer = JNIService.loadIndex(
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                KNNEngine.NMSLIB
            );
            assertNotEquals(0, pointer);

            for (float[] query : testData.queries) {
                KNNQueryResult[] results = JNIService.queryIndex(pointer, query, k, null, KNNEngine.NMSLIB, null, 0, null);
                assertEquals(k, results.length);
            }
        }
    }

    public void testQueryIndex_faiss_invalid_badPointer() {

        expectThrows(Exception.class, () -> JNIService.queryIndex(0L, new float[] {}, 0, null, KNNEngine.FAISS, null, 0, null));
    }

    public void testQueryIndex_faiss_invalid_nullQueryVector() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, pointer);

        expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, null, KNNEngine.FAISS, null, 0, null));
    }

    public void testQueryIndex_faiss_valid() throws IOException {

        int k = 10;
        int efSearch = 100;

        List<String> methods = ImmutableList.of(faissMethod);
        List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        for (String method : methods) {
            for (SpaceType spaceType : spaces) {
                Path tmpFile = createTempFile();
                JNIService.createIndex(
                    testData.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testData.indexData.getDimension(),
                    tmpFile.toAbsolutePath().toString(),
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    KNNEngine.FAISS
                );
                assertTrue(tmpFile.toFile().length() > 0);

                long pointer = JNIService.loadIndex(
                    tmpFile.toAbsolutePath().toString(),
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
                }
            }
        }
    }

    public void testQueryIndex_faiss_parentIds() throws IOException {

        int k = 100;
        int efSearch = 100;

        List<String> methods = ImmutableList.of(faissMethod);
        List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        int[] parentIds = toParentIdArray(testDataNested.indexData.docs);
        Map<Integer, Integer> idToParentIdMap = toIdToParentIdMap(testDataNested.indexData.docs);
        for (String method : methods) {
            for (SpaceType spaceType : spaces) {
                Path tmpFile = createTempFile();
                JNIService.createIndex(
                    testDataNested.indexData.docs,
                    testData.loadDataToMemoryAddress(),
                    testDataNested.indexData.getDimension(),
                    tmpFile.toAbsolutePath().toString(),
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    KNNEngine.FAISS
                );
                assertTrue(tmpFile.toFile().length() > 0);

                long pointer = JNIService.loadIndex(
                    tmpFile.toAbsolutePath().toString(),
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
                }
            }
        }
    }

    @SneakyThrows
    public void testQueryBinaryIndex_faiss_valid() {
        int k = 10;
        List<String> methods = ImmutableList.of(faissBinaryMethod);
        for (String method : methods) {
            Path tmpFile = createTempFile();
            long memoryAddr = testData.loadBinaryDataToMemoryAddress();
            JNIService.createIndex(
                testData.indexData.docs,
                memoryAddr,
                testData.indexData.getDimension(),
                tmpFile.toAbsolutePath().toString(),
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
            assertTrue(tmpFile.toFile().length() > 0);

            long pointer = JNIService.loadIndex(
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue()),
                KNNEngine.FAISS
            );
            assertNotEquals(0, pointer);

            for (byte[] query : testData.binaryQueries) {
                KNNQueryResult[] results = JNIService.queryBinaryIndex(pointer, query, k, null, KNNEngine.FAISS, null, 0, null);
                assertEquals(k, results.length);
            }
        }
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

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB
        );
        assertNotEquals(0, pointer);

        JNIService.free(pointer, KNNEngine.NMSLIB);
    }

    public void testFree_faiss_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, pointer);

        JNIService.free(pointer, KNNEngine.FAISS);
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
        Map<String, Object> parameters = KNNEngine.FAISS.getMethodAsMap(knnMethodContext);

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
        Map<String, Object> parameters = KNNEngine.FAISS.getMethodAsMap(knnMethodContext);

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
        knnMethodContext.getMethodComponentContext().setIndexVersion(Version.CURRENT);
        Map<String, Object> parameters = KNNEngine.FAISS.getMethodAsMap(knnMethodContext);

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

    public void testCreateIndexFromTemplate() throws IOException {

        long trainPointer1 = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer1);

        long trainPointer2;
        for (int i = 0; i < 10; i++) {
            trainPointer2 = JNIService.transferVectors(trainPointer1, testData.indexData.vectors);
            assertEquals(trainPointer1, trainPointer2);
        }

        SpaceType spaceType = SpaceType.L2;
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

        String description = knnMethodContext.getKnnEngine().getMethodAsMap(knnMethodContext).get(INDEX_DESCRIPTION_PARAMETER).toString();
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

        Path tmpFile1 = createTempFile();
        JNIService.createIndexFromTemplate(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile1.toAbsolutePath().toString(),
            faissIndex,
            ImmutableMap.of(INDEX_THREAD_QTY, 1),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile1.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile1.toAbsolutePath().toString(), Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, pointer);
    }

    @SneakyThrows
    public void testIndexLoad_whenStateIsShared_thenSucceed() {
        // Creates a single IVFPQ-l2 index. Then, we will configure a set of indices in memory in different ways to
        // ensure that everything is loaded properly and the results are consistent.
        int k = 10;
        int ivfNlist = 16;
        int pqM = 16;
        int pqCodeSize = 4;

        String indexIVFPQPath = createFaissIVFPQIndex(ivfNlist, pqM, pqCodeSize, SpaceType.L2);

        long indexIVFPQIndexTest1 = JNIService.loadIndex(indexIVFPQPath, Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, indexIVFPQIndexTest1);
        long indexIVFPQIndexTest2 = JNIService.loadIndex(indexIVFPQPath, Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, indexIVFPQIndexTest2);

        long sharedStateAddress = JNIService.initSharedIndexState(indexIVFPQIndexTest1, KNNEngine.FAISS);
        JNIService.setSharedIndexState(indexIVFPQIndexTest1, sharedStateAddress, KNNEngine.FAISS);
        JNIService.setSharedIndexState(indexIVFPQIndexTest2, sharedStateAddress, KNNEngine.FAISS);

        assertQueryResultsMatch(testData.queries, k, List.of(indexIVFPQIndexTest1, indexIVFPQIndexTest2));

        // Free the first test index 1. This will ensure that the shared state persists after index that initialized
        // shared state is gone.
        JNIService.free(indexIVFPQIndexTest1, KNNEngine.FAISS);

        long indexIVFPQIndexTest3 = JNIService.loadIndex(indexIVFPQPath, Collections.emptyMap(), KNNEngine.FAISS);
        assertNotEquals(0, indexIVFPQIndexTest3);

        JNIService.setSharedIndexState(indexIVFPQIndexTest3, sharedStateAddress, KNNEngine.FAISS);

        assertQueryResultsMatch(testData.queries, k, List.of(indexIVFPQIndexTest2, indexIVFPQIndexTest3));

        // Ensure everything gets freed
        JNIService.free(indexIVFPQIndexTest2, KNNEngine.FAISS);
        JNIService.free(indexIVFPQIndexTest3, KNNEngine.FAISS);
        JNIService.freeSharedIndexState(sharedStateAddress, KNNEngine.FAISS);
    }

    @SneakyThrows
    public void testIsIndexIVFPQL2() {
        long dummyAddress = 0;
        assertFalse(JNIService.isSharedIndexStateRequired(dummyAddress, KNNEngine.NMSLIB));

        String faissIVFPQL2Index = createFaissIVFPQIndex(16, 16, 4, SpaceType.L2);
        long faissIVFPQL2Address = JNIService.loadIndex(faissIVFPQL2Index, Collections.emptyMap(), KNNEngine.FAISS);
        assertTrue(JNIService.isSharedIndexStateRequired(faissIVFPQL2Address, KNNEngine.FAISS));
        JNIService.free(faissIVFPQL2Address, KNNEngine.FAISS);

        String faissIVFPQIPIndex = createFaissIVFPQIndex(16, 16, 4, SpaceType.INNER_PRODUCT);
        long faissIVFPQIPAddress = JNIService.loadIndex(faissIVFPQIPIndex, Collections.emptyMap(), KNNEngine.FAISS);
        assertFalse(JNIService.isSharedIndexStateRequired(faissIVFPQIPAddress, KNNEngine.FAISS));
        JNIService.free(faissIVFPQIPAddress, KNNEngine.FAISS);

        String faissHNSWIndex = createFaissHNSWIndex(SpaceType.L2);
        long faissHNSWAddress = JNIService.loadIndex(faissHNSWIndex, Collections.emptyMap(), KNNEngine.FAISS);
        assertFalse(JNIService.isSharedIndexStateRequired(faissHNSWAddress, KNNEngine.FAISS));
        JNIService.free(faissHNSWAddress, KNNEngine.FAISS);
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

    private String createFaissIVFPQIndex(int ivfNlist, int pqM, int pqCodeSize, SpaceType spaceType) throws IOException {
        long trainPointer = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer);

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

        String description = knnMethodContext.getKnnEngine().getMethodAsMap(knnMethodContext).get(INDEX_DESCRIPTION_PARAMETER).toString();
        Map<String, Object> parameters = ImmutableMap.of(
            INDEX_DESCRIPTION_PARAMETER,
            description,
            KNNConstants.SPACE_TYPE,
            spaceType.getValue()
        );

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer, KNNEngine.FAISS);

        assertNotEquals(0, faissIndex.length);
        JNICommons.freeVectorData(trainPointer);
        Path tmpFile = createTempFile();
        JNIService.createIndexFromTemplate(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            faissIndex,
            ImmutableMap.of(INDEX_THREAD_QTY, 1),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);

        return tmpFile.toAbsolutePath().toString();
    }

    private String createFaissHNSWIndex(SpaceType spaceType) throws IOException {
        Path tmpFile = createTempFile();
        JNIService.createIndex(
            testData.indexData.docs,
            testData.loadDataToMemoryAddress(),
            testData.indexData.getDimension(),
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, spaceType.getValue()),
            KNNEngine.FAISS
        );
        assertTrue(tmpFile.toFile().length() > 0);
        return tmpFile.toAbsolutePath().toString();
    }
}
