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
import org.junit.BeforeClass;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QTY;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;

public class JNIServiceTests extends KNNTestCase {

    static TestUtils.TestData testData;
    private String faissMethod = "HNSW32,Flat";
    private String faissNSGMethod = "NSG64,Flat";

    @BeforeClass
    public static void setUpClass() throws IOException {
        URL testIndexVectors = JNIServiceTests.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = JNIServiceTests.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    public void testCreateIndex_invalid_engineNotSupported() {
        expectThrows(
            IllegalArgumentException.class,
            () -> JNIService.createIndex(
                new int[] {},
                new float[][] {},
                "test",
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                "invalid-engine"
            )
        );
    }

    public void testCreateIndex_invalid_engineNull() {
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                new int[] {},
                new float[][] {},
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
                testData.indexData.vectors,
                "something",
                Collections.emptyMap(),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_vectorDocIDMismatch() throws IOException {

        int[] docIds = new int[] { 1, 2, 3 };
        float[][] vectors1 = new float[][] { { 1, 2 }, { 3, 4 } };

        Path tmpFile1 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors1,
                tmpFile1.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );

        float[][] vectors2 = new float[][] { { 1, 2 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 8, 9 } };

        Path tmpFile2 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors2,
                tmpFile2.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_nullArgument() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                null,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                null,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                null,
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(docIds, vectors, tmpFile.toAbsolutePath().toString(), null, KNNEngine.NMSLIB.getName())
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                null
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_badSpace() throws IOException {

        int[] docIds = new int[] { 1 };
        float[][] vectors = new float[][] { { 2, 3 } };

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, "invalid"),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_inconsistentDimensions() throws IOException {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 2, 3, 4 } };

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testCreateIndex_nmslib_invalid_badParameterType() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};

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
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue(), KNNConstants.PARAMETERS, parametersMap),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testCreateIndex_nmslib_valid() throws IOException {

        for (SpaceType spaceType : KNNEngine.NMSLIB.getMethod(KNNConstants.METHOD_HNSW).getSpaces()) {
            Path tmpFile = createTempFile();

            JNIService.createIndex(
                testData.indexData.docs,
                testData.indexData.vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                KNNEngine.NMSLIB.getName()
            );
            assertTrue(tmpFile.toFile().length() > 0);

            tmpFile = createTempFile();

            JNIService.createIndex(
                testData.indexData.docs,
                testData.indexData.vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(
                    KNNConstants.SPACE_TYPE,
                    spaceType.getValue(),
                    KNNConstants.HNSW_ALGO_EF_CONSTRUCTION,
                    14,
                    KNNConstants.METHOD_PARAMETER_M,
                    12
                ),
                KNNEngine.NMSLIB.getName()
            );
            assertTrue(tmpFile.toFile().length() > 0);
        }
    }

    public void testCreateIndex_faiss_invalid_noSpaceType() {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                "something",
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod),
                FAISS_NAME
            )
        );
    }

    public void testCreateIndex_faiss_invalid_vectorDocIDMismatch() throws IOException {

        int[] docIds = new int[] { 1, 2, 3 };
        float[][] vectors1 = new float[][] { { 1, 2 }, { 3, 4 } };

        Path tmpFile1 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors1,
                tmpFile1.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );

        float[][] vectors2 = new float[][] { { 1, 2 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 8, 9 } };

        Path tmpFile2 = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors2,
                tmpFile2.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );
    }

    public void testCreateIndex_faiss_invalid_null() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                null,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                null,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                null,
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );

        expectThrows(Exception.class, () -> JNIService.createIndex(docIds, vectors, tmpFile.toAbsolutePath().toString(), null, FAISS_NAME));

        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                null
            )
        );
    }

    public void testCreateIndex_faiss_invalid_invalidSpace() throws IOException {

        int[] docIds = new int[] { 1 };
        float[][] vectors = new float[][] { { 2, 3 } };

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, "invalid"),
                FAISS_NAME
            )
        );
    }

    public void testCreateIndex_faiss_invalid_inconsistentDimensions() throws IOException {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 2, 3, 4 } };

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );
    }

    public void testCreateIndex_faiss_invalid_noIndexDescription() throws IOException {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 2, 3, 4 } };

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );
    }

    public void testCreateIndex_faiss_invalid_invalidIndexDescription() throws IOException {

        int[] docIds = new int[] { 1, 2 };
        float[][] vectors = new float[][] { { 2, 3 }, { 2, 3, 4 } };

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, "invalid", KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                FAISS_NAME
            )
        );
    }

    public void testCreateIndex_faiss_invalid_invalidParameterType() throws IOException {

        int[] docIds = new int[] {};
        float[][] vectors = new float[][] {};

        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.createIndex(
                docIds,
                vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(
                    INDEX_DESCRIPTION_PARAMETER,
                    "IVF13",
                    KNNConstants.SPACE_TYPE,
                    SpaceType.L2.getValue(),
                    KNNConstants.PARAMETERS,
                    ImmutableMap.of(KNNConstants.METHOD_PARAMETER_NPROBES, "14")
                ),
                FAISS_NAME
            )
        );

    }

    public void testCreateIndex_faiss_valid() throws IOException {

        List<String> methods = ImmutableList.of(faissMethod, faissNSGMethod);
        List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        for (String method : methods) {
            for (SpaceType spaceType : spaces) {
                Path tmpFile1 = createTempFile();
                JNIService.createIndex(
                    testData.indexData.docs,
                    testData.indexData.vectors,
                    tmpFile1.toAbsolutePath().toString(),
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    FAISS_NAME
                );
                assertTrue(tmpFile1.toFile().length() > 0);
            }
        }
    }

    public void testLoadIndex_invalidEngine() {
        expectThrows(IllegalArgumentException.class, () -> JNIService.loadIndex("test", Collections.emptyMap(), "invalid-engine"));
    }

    public void testLoadIndex_nmslib_invalid_badSpaceType() {
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex("test", ImmutableMap.of(KNNConstants.SPACE_TYPE, "invalid"), KNNEngine.NMSLIB.getName())
        );
    }

    public void testLoadIndex_nmslib_invalid_noSpaceType() {
        expectThrows(Exception.class, () -> JNIService.loadIndex("test", Collections.emptyMap(), KNNEngine.NMSLIB.getName()));
    }

    public void testLoadIndex_nmslib_invalid_fileDoesNotExist() {
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(
                "invalid",
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testLoadIndex_nmslib_invalid_badFile() throws IOException {
        Path tmpFile = createTempFile();
        expectThrows(
            Exception.class,
            () -> JNIService.loadIndex(
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
                KNNEngine.NMSLIB.getName()
            )
        );
    }

    public void testLoadIndex_nmslib_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB.getName()
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB.getName()
        );
        assertNotEquals(0, pointer);
    }

    public void testLoadIndex_faiss_invalid_fileDoesNotExist() {
        expectThrows(Exception.class, () -> JNIService.loadIndex("invalid", Collections.emptyMap(), FAISS_NAME));
    }

    public void testLoadIndex_faiss_invalid_badFile() throws IOException {

        Path tmpFile = createTempFile();

        expectThrows(Exception.class, () -> JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), FAISS_NAME));
    }

    public void testLoadIndex_faiss_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            FAISS_NAME
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), FAISS_NAME);
        assertNotEquals(0, pointer);
    }

    public void testQueryIndex_invalidEngine() {
        expectThrows(IllegalArgumentException.class, () -> JNIService.queryIndex(0L, new float[] {}, 0, "invalid" + "-engine", null));
    }

    public void testQueryIndex_nmslib_invalid_badPointer() {

        expectThrows(Exception.class, () -> JNIService.queryIndex(0L, new float[] {}, 0, KNNEngine.NMSLIB.getName(), null));
    }

    public void testQueryIndex_nmslib_invalid_nullQueryVector() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB.getName()
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB.getName()
        );
        assertNotEquals(0, pointer);

        expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, KNNEngine.NMSLIB.getName(), null));
    }

    public void testQueryIndex_nmslib_valid() throws IOException {

        int k = 50;
        for (SpaceType spaceType : KNNEngine.NMSLIB.getMethod(KNNConstants.METHOD_HNSW).getSpaces()) {
            Path tmpFile = createTempFile();

            JNIService.createIndex(
                testData.indexData.docs,
                testData.indexData.vectors,
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                KNNEngine.NMSLIB.getName()
            );
            assertTrue(tmpFile.toFile().length() > 0);

            long pointer = JNIService.loadIndex(
                tmpFile.toAbsolutePath().toString(),
                ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                KNNEngine.NMSLIB.getName()
            );
            assertNotEquals(0, pointer);

            for (float[] query : testData.queries) {
                KNNQueryResult[] results = JNIService.queryIndex(pointer, query, k, KNNEngine.NMSLIB.getName(), null);
                assertEquals(k, results.length);
            }
        }
    }

    public void testQueryIndex_faiss_invalid_badPointer() {

        expectThrows(Exception.class, () -> JNIService.queryIndex(0L, new float[] {}, 0, FAISS_NAME, null));
    }

    public void testQueryIndex_faiss_invalid_nullQueryVector() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            FAISS_NAME
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), FAISS_NAME);
        assertNotEquals(0, pointer);

        expectThrows(Exception.class, () -> JNIService.queryIndex(pointer, null, 10, FAISS_NAME, null));
    }

    public void testQueryIndex_faiss_valid() throws IOException {

        int k = 10;

        List<String> methods = ImmutableList.of(faissMethod, faissNSGMethod);
        List<SpaceType> spaces = ImmutableList.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        for (String method : methods) {
            for (SpaceType spaceType : spaces) {
                Path tmpFile = createTempFile();
                JNIService.createIndex(
                    testData.indexData.docs,
                    testData.indexData.vectors,
                    tmpFile.toAbsolutePath().toString(),
                    ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, method, KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    FAISS_NAME
                );
                assertTrue(tmpFile.toFile().length() > 0);

                long pointer = JNIService.loadIndex(
                    tmpFile.toAbsolutePath().toString(),
                    ImmutableMap.of(KNNConstants.SPACE_TYPE, spaceType.getValue()),
                    FAISS_NAME
                );
                assertNotEquals(0, pointer);

                for (float[] query : testData.queries) {
                    KNNQueryResult[] results = JNIService.queryIndex(pointer, query, k, FAISS_NAME, null);
                    assertEquals(k, results.length);
                }

                // Filter will result in no ids, But NSG not supported Search params
                if (!method.contains(faissNSGMethod)) {
                    for (float[] query : testData.queries) {
                        KNNQueryResult[] results = JNIService.queryIndex(pointer, query, k, FAISS_NAME, new int[] { 0 });
                        assertEquals(0, results.length);
                    }
                }
            }
        }
    }

    public void testFree_invalidEngine() {
        expectThrows(IllegalArgumentException.class, () -> JNIService.free(0L, "invalid-engine"));
    }

    public void testFree_nmslib_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB.getName()
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            KNNEngine.NMSLIB.getName()
        );
        assertNotEquals(0, pointer);

        JNIService.free(pointer, KNNEngine.NMSLIB.getName());
    }

    public void testFree_faiss_valid() throws IOException {

        Path tmpFile = createTempFile();

        JNIService.createIndex(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile.toAbsolutePath().toString(),
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, faissMethod, KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()),
            FAISS_NAME
        );
        assertTrue(tmpFile.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile.toAbsolutePath().toString(), Collections.emptyMap(), FAISS_NAME);
        assertNotEquals(0, pointer);

        JNIService.free(pointer, FAISS_NAME);
    }

    public void testTransferVectors() {
        long trainPointer1 = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer1);

        long trainPointer2;
        for (int i = 0; i < 10; i++) {
            trainPointer2 = JNIService.transferVectors(trainPointer1, testData.indexData.vectors);
            assertEquals(trainPointer1, trainPointer2);
        }

        JNIService.freeVectors(trainPointer1);
    }

    public void testTrain() {

        long trainPointer1 = JNIService.transferVectors(0, testData.indexData.vectors);
        assertNotEquals(0, trainPointer1);

        long trainPointer2;
        for (int i = 0; i < 10; i++) {
            trainPointer2 = JNIService.transferVectors(trainPointer1, testData.indexData.vectors);
            assertEquals(trainPointer1, trainPointer2);
        }

        Map<String, Object> parameters = ImmutableMap.of(
            INDEX_DESCRIPTION_PARAMETER,
            "IVF16,PQ4",
            KNNConstants.SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer1, FAISS_NAME);

        assertNotEquals(0, faissIndex.length);
        JNIService.freeVectors(trainPointer1);
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

        byte[] faissIndex = JNIService.trainIndex(parameters, 128, trainPointer1, FAISS_NAME);

        assertNotEquals(0, faissIndex.length);
        JNIService.freeVectors(trainPointer1);

        Path tmpFile1 = createTempFile();
        JNIService.createIndexFromTemplate(
            testData.indexData.docs,
            testData.indexData.vectors,
            tmpFile1.toAbsolutePath().toString(),
            faissIndex,
            ImmutableMap.of(INDEX_THREAD_QTY, 1),
            FAISS_NAME
        );
        assertTrue(tmpFile1.toFile().length() > 0);

        long pointer = JNIService.loadIndex(tmpFile1.toAbsolutePath().toString(), Collections.emptyMap(), FAISS_NAME);
        assertNotEquals(0, pointer);
    }
}
