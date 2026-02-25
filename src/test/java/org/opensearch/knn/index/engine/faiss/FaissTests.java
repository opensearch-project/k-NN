/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import lombok.SneakyThrows;
import org.opensearch.Version;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class FaissTests extends KNNTestCase {

    public void testGetKNNLibraryIndexingContext_whenMethodIsHNSWFlat_thenCreateCorrectIndexDescription() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();

        int mParam = 65;
        String expectedIndexDescription = String.format(Locale.ROOT, "HNSW%d,Flat", mParam);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mParam)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);

        Map<String, Object> map = Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testGetKNNLibraryIndexingContext_whenMethodIsHNSWPQ_thenCreateCorrectIndexDescription() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int hnswMParam = 65;
        int pqMParam = 17;
        String expectedIndexDescription = String.format(Locale.ROOT, "HNSW%d,PQ%d", hnswMParam, pqMParam);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, hnswMParam)
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

        Map<String, Object> map = Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    @SneakyThrows
    public void testGetKNNLibraryIndexingContext_whenMethodIsHNSWSQFP16_thenCreateCorrectIndexDescription() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int hnswMParam = 65;
        String expectedIndexDescription = String.format(Locale.ROOT, "HNSW%d,SQfp16", hnswMParam);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, hnswMParam)
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

        Map<String, Object> map = Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testGetKNNLibraryIndexingContext_whenMethodIsIVFFlat_thenCreateCorrectIndexDescription() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int nlists = 88;
        String expectedIndexDescription = String.format(Locale.ROOT, "IVF%d,Flat", nlists);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, nlists)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);

        Map<String, Object> map = Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testGetKNNLibraryIndexingContext_whenMethodIsIVFPQ_thenCreateCorrectIndexDescription() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int ivfNlistsParam = 88;
        int pqMParam = 17;
        int pqCodeSizeParam = 53;
        String expectedIndexDescription = String.format(Locale.ROOT, "IVF%d,PQ%dx%d", ivfNlistsParam, pqMParam, pqCodeSizeParam);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, ivfNlistsParam)
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

        Map<String, Object> map = Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    @SneakyThrows
    public void testGetKNNLibraryIndexingContext_whenMethodIsIVFSQFP16_thenCreateCorrectIndexDescription() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int nlists = 88;
        String expectedIndexDescription = String.format(Locale.ROOT, "IVF%d,SQfp16", nlists);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, nlists)
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

        Map<String, Object> map = Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    @SneakyThrows
    public void testGetKNNLibraryIndexingContext_whenMethodIsHNSWWithQFrame_thenCreateCorrectConfig() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int m = 88;
        String expectedIndexDescription = "BHNSW" + m + ",Flat";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, m)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, QFrameBitEncoder.NAME)
            .startObject(PARAMETERS)
            .field(QFrameBitEncoder.BITCOUNT_PARAM, 4)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        KNNLibraryIndexingContext knnLibraryIndexingContext = Faiss.INSTANCE.getKNNLibraryIndexingContext(
            knnMethodContext,
            knnMethodConfigContext
        );
        Map<String, Object> map = knnLibraryIndexingContext.getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build(),
            knnLibraryIndexingContext.getQuantizationConfig()
        );
    }

    @SneakyThrows
    public void testGetKNNLibraryIndexingContext_whenMethodIsIVFWithQFrame_thenCreateCorrectConfig() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        int nlist = 88;
        String expectedIndexDescription = "BIVF" + nlist + ",Flat";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, nlist)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, QFrameBitEncoder.NAME)
            .startObject(PARAMETERS)
            .field(QFrameBitEncoder.BITCOUNT_PARAM, 2)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        KNNLibraryIndexingContext knnLibraryIndexingContext = Faiss.INSTANCE.getKNNLibraryIndexingContext(
            knnMethodContext,
            knnMethodConfigContext
        );
        Map<String, Object> map = knnLibraryIndexingContext.getLibraryParameters();

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build(),
            knnLibraryIndexingContext.getQuantizationConfig()
        );
    }

    public void testMethodAsMapBuilder() throws IOException {
        String methodName = "test-method";
        String methodDescription = "test-description";
        String parameter1 = "test-parameter-1";
        Integer value1 = 10;
        Integer defaultValue1 = 1;
        String parameter2 = "test-parameter-2";
        Integer value2 = 15;
        Integer defaultValue2 = 2;
        String parameter3 = "test-parameter-3";
        Integer defaultValue3 = 3;
        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .addParameter(parameter1, new Parameter.IntegerParameter(parameter1, defaultValue1, (value, context) -> value > 0))
            .addParameter(parameter2, new Parameter.IntegerParameter(parameter2, defaultValue2, (value, context) -> value > 0))
            .addParameter(parameter3, new Parameter.IntegerParameter(parameter3, defaultValue3, (value, context) -> value > 0))
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field(parameter1, value1)
            .field(parameter2, value2)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext methodComponentContext = MethodComponentContext.parse(in);

        Map<String, Object> expectedParametersMap = new HashMap<>(methodComponentContext.getParameters());
        expectedParametersMap.put(parameter3, defaultValue3);
        Map<String, Object> expectedMap = new HashMap<>();
        expectedMap.put(PARAMETERS, expectedParametersMap);
        expectedMap.put(NAME, methodName);
        expectedMap.put(INDEX_DESCRIPTION_PARAMETER, methodDescription + value1);
        KNNLibraryIndexingContext expectedKNNMethodContext = KNNLibraryIndexingContextImpl.builder().parameters(expectedMap).build();

        KNNLibraryIndexingContext actualKNNLibraryIndexingContext = MethodAsMapBuilder.builder(
            methodDescription,
            methodComponent,
            methodComponentContext,
            KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build()
        ).addParameter(parameter1, "", "").build();

        assertEquals(expectedKNNMethodContext.getQuantizationConfig(), actualKNNLibraryIndexingContext.getQuantizationConfig());
        assertEquals(expectedKNNMethodContext.getLibraryParameters(), actualKNNLibraryIndexingContext.getLibraryParameters());
        assertEquals(expectedKNNMethodContext.getPerDimensionProcessor(), actualKNNLibraryIndexingContext.getPerDimensionProcessor());
        assertEquals(expectedKNNMethodContext.getPerDimensionValidator(), actualKNNLibraryIndexingContext.getPerDimensionValidator());
        assertEquals(expectedKNNMethodContext.getVectorValidator(), actualKNNLibraryIndexingContext.getVectorValidator());
    }

    /**
     * Test that scoreToRadialThreshold correctly converts Lucene scores to Faiss distances for INNER_PRODUCT.
     * This test validates the fix for the bug where min_score filters were returning results below the threshold.
     *
     * The bug scenario:
     * - Query vector: [1.0, 0.0], Indexed vector: [0.0, 1.0] (orthogonal)
     * - Inner product = 0
     * - Lucene score = 1.0 (from scoreTranslation: 0 <= 0 ? 1/(1-0) : 0+1 = 1.0)
     * - User sets min_score = 1.4
     * - Expected: orthogonal vector should be filtered out (score 1.0 < 1.4)
     * - Bug: Faiss was receiving negative distance threshold, returning the vector incorrectly
     */
    public void testScoreToRadialThreshold_whenInnerProduct_thenCorrectlyConvertsScoreToDistance() {
        // Test case 1: Score > 1 (positive inner product region)
        // Lucene score 1.4 should map to inner product distance 0.4
        // Because: distance + 1 = 1.4 => distance = 0.4
        float score1 = 1.4f;
        float expectedDistance1 = 0.4f;
        float actualDistance1 = Faiss.INSTANCE.scoreToRadialThreshold(score1, org.opensearch.knn.index.SpaceType.INNER_PRODUCT);
        assertEquals("Score 1.4 should convert to distance 0.4 (inner product >= 0.4)", expectedDistance1, actualDistance1, 1e-6);

        // Test case 2: Score = 1.0 (boundary case, inner product = 0)
        // Lucene score 1.0 should map to inner product distance 0.0
        float score2 = 1.0f;
        float expectedDistance2 = 0.0f;
        float actualDistance2 = Faiss.INSTANCE.scoreToRadialThreshold(score2, org.opensearch.knn.index.SpaceType.INNER_PRODUCT);
        assertEquals(
            "Score 1.0 should convert to distance 0.0 (inner product = 0, orthogonal vectors)",
            expectedDistance2,
            actualDistance2,
            1e-6
        );

        // Test case 3: Score < 1 (negative inner product region)
        // Lucene score 0.5 should map to inner product distance -1.0
        // Because: 1/(1-distance) = 0.5 => 1-distance = 2 => distance = -1.0
        float score3 = 0.5f;
        float expectedDistance3 = -1.0f;
        float actualDistance3 = Faiss.INSTANCE.scoreToRadialThreshold(score3, org.opensearch.knn.index.SpaceType.INNER_PRODUCT);
        assertEquals("Score 0.5 should convert to distance -1.0 (inner product = -1.0)", expectedDistance3, actualDistance3, 1e-6);

        // Test case 4: Score = 2.0 (high positive inner product)
        // Lucene score 2.0 should map to inner product distance 1.0
        float score4 = 2.0f;
        float expectedDistance4 = 1.0f;
        float actualDistance4 = Faiss.INSTANCE.scoreToRadialThreshold(score4, org.opensearch.knn.index.SpaceType.INNER_PRODUCT);
        assertEquals("Score 2.0 should convert to distance 1.0 (inner product = 1.0)", expectedDistance4, actualDistance4, 1e-6);

        // Test case 5: Score = 0.667 (negative inner product)
        // Lucene score 0.667 should map to inner product distance -0.5
        // Because: 1/(1-distance) = 0.667 => 1-distance = 1.5 => distance = -0.5
        float score5 = 2.0f / 3.0f; // 0.667
        float expectedDistance5 = -0.5f;
        float actualDistance5 = Faiss.INSTANCE.scoreToRadialThreshold(score5, org.opensearch.knn.index.SpaceType.INNER_PRODUCT);
        assertEquals("Score 0.667 should convert to distance -0.5 (inner product = -0.5)", expectedDistance5, actualDistance5, 1e-6);
    }

    /**
     * Test the round-trip conversion: distance -> score -> distance for INNER_PRODUCT.
     * This ensures scoreToRadialThreshold is the proper inverse of distanceToRadialThreshold.
     */
    public void testInnerProductRoundTripConversion_whenDistanceToScoreToDistance_thenReturnsOriginalDistance() {
        // Test positive inner products
        float[] testDistances = { 0.0f, 0.4f, 1.0f, 5.5f, -0.5f, -1.0f, -2.0f };

        for (float originalDistance : testDistances) {
            // Convert distance to Lucene score
            float luceneScore = org.opensearch.knn.index.engine.lucene.Lucene.INSTANCE.distanceToRadialThreshold(
                originalDistance,
                org.opensearch.knn.index.SpaceType.INNER_PRODUCT
            );

            // Convert Lucene score back to Faiss distance
            float convertedDistance = Faiss.INSTANCE.scoreToRadialThreshold(luceneScore, org.opensearch.knn.index.SpaceType.INNER_PRODUCT);

            assertEquals(
                String.format("Round-trip conversion failed for distance %.2f (score was %.4f)", originalDistance, luceneScore),
                originalDistance,
                convertedDistance,
                1e-5
            );
        }
    }
}
