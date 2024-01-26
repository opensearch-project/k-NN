/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.SneakyThrows;
import org.opensearch.Version;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.Parameter;

import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQFP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
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

    public void testGetMethodAsMap_whenMethodIsHNSWFlat_thenCreateCorrectIndexDescription() throws IOException {
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
        knnMethodContext.getMethodComponentContext().setIndexVersion(Version.CURRENT);

        Map<String, Object> map = Faiss.INSTANCE.getMethodAsMap(knnMethodContext);

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testGetMethodAsMap_whenMethodIsHNSWPQ_thenCreateCorrectIndexDescription() throws IOException {
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
        knnMethodContext.getMethodComponentContext().setIndexVersion(Version.CURRENT);

        Map<String, Object> map = Faiss.INSTANCE.getMethodAsMap(knnMethodContext);

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    @SneakyThrows
    public void testGetMethodAsMap_whenMethodIsHNSWSQFP16_thenCreateCorrectIndexDescription() {
        int hnswMParam = 65;
        String expectedIndexDescription = String.format(Locale.ROOT, "HNSW%d,SQfp16", hnswMParam);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, hnswMParam)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQFP16)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        knnMethodContext.getMethodComponentContext().setIndexVersion(Version.CURRENT);

        Map<String, Object> map = Faiss.INSTANCE.getMethodAsMap(knnMethodContext);

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testGetMethodAsMap_whenMethodIsIVFFlat_thenCreateCorrectIndexDescription() throws IOException {
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

        Map<String, Object> map = Faiss.INSTANCE.getMethodAsMap(knnMethodContext);

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testGetMethodAsMap_whenMethodIsIVFPQ_thenCreateCorrectIndexDescription() throws IOException {
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

        Map<String, Object> map = Faiss.INSTANCE.getMethodAsMap(knnMethodContext);

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
    }

    @SneakyThrows
    public void testGetMethodAsMap_whenMethodIsIVFSQFP16_thenCreateCorrectIndexDescription() {
        int nlists = 88;
        String expectedIndexDescription = String.format(Locale.ROOT, "IVF%d,SQfp16", nlists);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, nlists)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQFP16)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);

        Map<String, Object> map = Faiss.INSTANCE.getMethodAsMap(knnMethodContext);

        assertTrue(map.containsKey(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(expectedIndexDescription, map.get(INDEX_DESCRIPTION_PARAMETER));
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
            .addParameter(parameter1, new Parameter.IntegerParameter(parameter1, defaultValue1, value -> value > 0))
            .addParameter(parameter2, new Parameter.IntegerParameter(parameter2, defaultValue2, value -> value > 0))
            .addParameter(parameter3, new Parameter.IntegerParameter(parameter3, defaultValue3, value -> value > 0))
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
        expectedParametersMap.remove(parameter1);
        Map<String, Object> expectedMap = new HashMap<>();
        expectedMap.put(PARAMETERS, expectedParametersMap);
        expectedMap.put(NAME, methodName);
        expectedMap.put(INDEX_DESCRIPTION_PARAMETER, methodDescription + value1);

        Map<String, Object> methodAsMap = Faiss.MethodAsMapBuilder.builder(methodDescription, methodComponent, methodComponentContext)
            .addParameter(parameter1, "", "")
            .build();

        assertEquals(expectedMap, methodAsMap);
    }

}
