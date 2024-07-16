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

package org.opensearch.knn.index;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;
import com.google.common.collect.ImmutableMap;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;

public class KNNMethodContextTests extends KNNTestCase {

    /**
     * Test reading from and writing to streams
     */
    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        String name = "test-name";
        Map<String, Object> parameters = ImmutableMap.of("test-p-1", 10, "test-p-2", "string-p");

        MethodComponentContext originalMethodComponent = new MethodComponentContext(name, parameters);

        KNNMethodContext original = new KNNMethodContext(knnEngine, spaceType, originalMethodComponent);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        original.writeTo(streamOutput);

        KNNMethodContext copy = new KNNMethodContext(streamOutput.bytes().streamInput());

        assertEquals(original, copy);
    }

    /**
     * Test method component getter
     */
    public void testGetMethodComponent() {
        MethodComponentContext methodComponent = new MethodComponentContext("test-method", Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, methodComponent);
        assertEquals(methodComponent, knnMethodContext.getMethodComponentContext());
    }

    /**
     * Test engine getter
     */
    public void testGetEngine() {
        MethodComponentContext methodComponent = new MethodComponentContext("test-method", Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, methodComponent);
        assertEquals(KNNEngine.DEFAULT, knnMethodContext.getKnnEngine());
    }

    /**
     * Test spaceType getter
     */
    public void testGetSpaceType() {
        MethodComponentContext methodComponent = new MethodComponentContext("test-method", Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.L1, methodComponent);
        assertEquals(SpaceType.L1, knnMethodContext.getSpaceType());
    }

    /**
     * Test KNNMethodContext validation
     */
    public void testValidate() {
        // Check valid default - this should not throw any exception
        assertNull(KNNMethodContext.getDefault().validate());

        // Check a valid nmslib method
        MethodComponentContext hnswMethod = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.NMSLIB, SpaceType.L2, hnswMethod);
        assertNull(knnMethodContext.validate());

        // Check invalid parameter nmslib
        hnswMethod = new MethodComponentContext(METHOD_HNSW, ImmutableMap.of("invalid", 111));
        KNNMethodContext knnMethodContext1 = new KNNMethodContext(KNNEngine.NMSLIB, SpaceType.L2, hnswMethod);
        assertNotNull(knnMethodContext1.validate());

        // Check invalid method nmslib
        MethodComponentContext invalidMethod = new MethodComponentContext("invalid", Collections.emptyMap());
        KNNMethodContext knnMethodContext2 = new KNNMethodContext(KNNEngine.NMSLIB, SpaceType.L2, invalidMethod);
        expectThrows(IllegalArgumentException.class, knnMethodContext2::validate);
    }

    /**
     * Test KNNMethodContext requires training method
     */
    public void testRequiresTraining() {

        // Check for NMSLIB
        MethodComponentContext hnswMethod = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.NMSLIB, SpaceType.L2, hnswMethod);
        assertFalse(knnMethodContext.isTrainingRequired());

        // Check for FAISS not required
        hnswMethod = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, hnswMethod);
        assertFalse(knnMethodContext.isTrainingRequired());

        // Check FAISS required
        MethodComponentContext pq = new MethodComponentContext(ENCODER_PQ, Collections.emptyMap());

        MethodComponentContext hnswMethodPq = new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_ENCODER_PARAMETER, pq));
        knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, hnswMethodPq);
        assertTrue(knnMethodContext.isTrainingRequired());

        MethodComponentContext ivfMethod = new MethodComponentContext(METHOD_IVF, Collections.emptyMap());
        knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, ivfMethod);
        assertTrue(knnMethodContext.isTrainingRequired());

        MethodComponentContext ivfMethodPq = new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_ENCODER_PARAMETER, pq));
        knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, ivfMethodPq);
        assertTrue(knnMethodContext.isTrainingRequired());
    }

    public void testEstimateOverheadInKB_whenMethodIsHNSWFlatNmslib_thenSizeIsExpectedValue() {
        // For HNSW no encoding we expect 0
        MethodComponentContext hnswMethod = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.NMSLIB, SpaceType.L2, hnswMethod);
        assertEquals(0, knnMethodContext.estimateOverheadInKB(1000));

    }

    public void testEstimateOverheadInKB_whenMethodIsHNSWFlatFaiss_thenSizeIsExpectedValue() {
        // For HNSW no encoding we expect 0
        MethodComponentContext hnswMethod = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.INNER_PRODUCT, hnswMethod);
        assertEquals(0, knnMethodContext.estimateOverheadInKB(168));

    }

    public void testEstimateOverheadInKB_whenMethodIsHNSWPQFaiss_thenSizeIsExpectedValue() {
        int dimension = 768;
        int codeSize = ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;

        // For HNSWPQ, we expect 4 * d * 2^code_size / 1024 + 1
        int expectedHnswPq = 4 * dimension * (1 << codeSize) / BYTES_PER_KILOBYTES + 1;

        MethodComponentContext pqMethodContext = new MethodComponentContext(ENCODER_PQ, ImmutableMap.of());

        MethodComponentContext hnswMethodPq = new MethodComponentContext(
            METHOD_HNSW,
            ImmutableMap.of(METHOD_ENCODER_PARAMETER, pqMethodContext)
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, hnswMethodPq);
        assertEquals(expectedHnswPq, knnMethodContext.estimateOverheadInKB(dimension));
    }

    public void testEstimateOverheadInKB_whenMethodIsIVFFlatFaiss_thenSizeIsExpectedValue() {
        // For IVF, we expect 4 * nlist * d / 1024 + 1
        int dimension = 768;
        int nlists = 1024;
        int expectedIvf = 4 * nlists * dimension / BYTES_PER_KILOBYTES + 1;

        MethodComponentContext ivfMethod = new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists));
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, ivfMethod);
        assertEquals(expectedIvf, knnMethodContext.estimateOverheadInKB(dimension));
    }

    public void testEstimateOverheadInKB_whenMethodIsIVFPQFaiss_thenSizeIsExpectedValue() {
        int dimension = 768;
        int nlists = 1024;
        int expectedIvf = 4 * nlists * dimension / BYTES_PER_KILOBYTES + 1;

        // For IVFPQ twe expect 4 * nlist * d / 1024 + 1 + 4 * d * 2^code_size / 1024 + 1
        int codeSize = 16;
        int expectedFromPq = 4 * dimension * (1 << codeSize) / BYTES_PER_KILOBYTES + 1;
        int expectedIvfPq = expectedIvf + expectedFromPq;

        MethodComponentContext pqMethodContext = new MethodComponentContext(
            ENCODER_PQ,
            ImmutableMap.of(ENCODER_PARAMETER_PQ_CODE_SIZE, codeSize)
        );

        MethodComponentContext ivfMethodPq = new MethodComponentContext(
            METHOD_IVF,
            ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists, METHOD_ENCODER_PARAMETER, pqMethodContext)
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, ivfMethodPq);
        assertEquals(expectedIvfPq, knnMethodContext.estimateOverheadInKB(dimension));
    }

    /**
     * Test context method parsing when input is invalid
     */
    public void testParse_invalid() throws IOException {
        // Invalid input type
        Integer invalidIn = 12;
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(invalidIn));

        // Invalid engine type
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().field(KNN_ENGINE, 0).endObject();

        final Map<String, Object> in0 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in0));

        // Invalid engine name
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(KNN_ENGINE, "invalid").endObject();

        final Map<String, Object> in1 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in1));

        // Invalid space type
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(METHOD_PARAMETER_SPACE_TYPE, 0).endObject();

        final Map<String, Object> in2 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in2));

        // Invalid space name
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(METHOD_PARAMETER_SPACE_TYPE, "invalid").endObject();

        final Map<String, Object> in3 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in3));

        // Invalid name not set
        xContentBuilder = XContentFactory.jsonBuilder().startObject().endObject();
        final Map<String, Object> in4 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in4));

        // Invalid name type
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, 13).endObject();

        final Map<String, Object> in5 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in5));

        // Invalid parameter type
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(PARAMETERS, 13).endObject();

        final Map<String, Object> in6 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> KNNMethodContext.parse(in6));

        // Invalid key
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field("invalid", 12).endObject();
        Map<String, Object> in7 = xContentBuilderToMap(xContentBuilder);
        expectThrows(MapperParsingException.class, () -> MethodComponentContext.parse(in7));
    }

    /**
     * Test context method parsing when parameters are set to null
     */
    public void testParse_nullParameters() throws IOException {
        String methodName = "test-method";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(PARAMETERS, (String) null)
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        assertTrue(knnMethodContext.getMethodComponentContext().getParameters().isEmpty());
    }

    /**
     * Test context method parsing when input is valid
     */
    public void testParse_valid() throws IOException {
        // Simple method with only name set
        String methodName = "test-method";

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, methodName).endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);

        assertEquals(KNNEngine.DEFAULT, knnMethodContext.getKnnEngine());
        assertEquals(SpaceType.UNDEFINED, knnMethodContext.getSpaceType());
        assertEquals(methodName, knnMethodContext.getMethodComponentContext().getName());
        assertTrue(knnMethodContext.getMethodComponentContext().getParameters().isEmpty());

        // Method with parameters
        String methodParameterKey1 = "p-1";
        String methodParameterValue1 = "v-1";
        String methodParameterKey2 = "p-2";
        Integer methodParameterValue2 = 27;

        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field(methodParameterKey1, methodParameterValue1)
            .field(methodParameterKey2, methodParameterValue2)
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        knnMethodContext = KNNMethodContext.parse(in);

        assertEquals(methodParameterValue1, knnMethodContext.getMethodComponentContext().getParameters().get(methodParameterKey1));
        assertEquals(methodParameterValue2, knnMethodContext.getMethodComponentContext().getParameters().get(methodParameterKey2));

        // Method with parameter that is a method context paramet

        // Parameter that is itself a MethodComponentContext
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .startObject(methodParameterKey1)
            .field(NAME, methodParameterValue1)
            .endObject()
            .field(methodParameterKey2, methodParameterValue2)
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        knnMethodContext = KNNMethodContext.parse(in);

        assertTrue(knnMethodContext.getMethodComponentContext().getParameters().get(methodParameterKey1) instanceof MethodComponentContext);
        assertEquals(
            methodParameterValue1,
            ((MethodComponentContext) knnMethodContext.getMethodComponentContext().getParameters().get(methodParameterKey1)).getName()
        );
        assertEquals(methodParameterValue2, knnMethodContext.getMethodComponentContext().getParameters().get(methodParameterKey2));
    }

    /**
     * Test toXContent method
     */
    public void testToXContent() throws IOException {
        String methodName = "test-method";
        String spaceType = SpaceType.L2.getValue();
        String knnEngine = KNNEngine.DEFAULT.getName();
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
            .field(KNN_ENGINE, knnEngine)
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder = knnMethodContext.toXContent(builder, ToXContent.EMPTY_PARAMS).endObject();

        Map<String, Object> out = xContentBuilderToMap(builder);
        assertEquals(methodName, out.get(NAME));
        assertEquals(spaceType, out.get(METHOD_PARAMETER_SPACE_TYPE));
        assertEquals(knnEngine, out.get(KNN_ENGINE));
    }

    public void testEquals() {
        SpaceType spaceType1 = SpaceType.L1;
        SpaceType spaceType2 = SpaceType.L2;
        String name1 = "name1";
        String name2 = "name2";
        Map<String, Object> parameters1 = ImmutableMap.of("param1", "v1", "param2", 18);

        MethodComponentContext methodComponentContext1 = new MethodComponentContext(name1, parameters1);
        MethodComponentContext methodComponentContext2 = new MethodComponentContext(name2, parameters1);

        KNNMethodContext methodContext1 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType1, methodComponentContext1);
        KNNMethodContext methodContext2 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType1, methodComponentContext1);
        KNNMethodContext methodContext3 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType1, methodComponentContext2);
        KNNMethodContext methodContext4 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType2, methodComponentContext1);
        KNNMethodContext methodContext5 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType2, methodComponentContext2);

        assertNotEquals(methodContext1, null);
        assertEquals(methodContext1, methodContext1);
        assertEquals(methodContext1, methodContext2);
        assertNotEquals(methodContext1, methodContext3);
        assertNotEquals(methodContext1, methodContext4);
        assertNotEquals(methodContext1, methodContext5);
    }

    public void testHashCode() {
        SpaceType spaceType1 = SpaceType.L1;
        SpaceType spaceType2 = SpaceType.L2;
        String name1 = "name1";
        String name2 = "name2";
        Map<String, Object> parameters1 = ImmutableMap.of("param1", "v1", "param2", 18);

        MethodComponentContext methodComponentContext1 = new MethodComponentContext(name1, parameters1);
        MethodComponentContext methodComponentContext2 = new MethodComponentContext(name2, parameters1);

        KNNMethodContext methodContext1 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType1, methodComponentContext1);
        KNNMethodContext methodContext2 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType1, methodComponentContext1);
        KNNMethodContext methodContext3 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType1, methodComponentContext2);
        KNNMethodContext methodContext4 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType2, methodComponentContext1);
        KNNMethodContext methodContext5 = new KNNMethodContext(KNNEngine.DEFAULT, spaceType2, methodComponentContext2);

        assertEquals(methodContext1.hashCode(), methodContext1.hashCode());
        assertEquals(methodContext1.hashCode(), methodContext2.hashCode());
        assertNotEquals(methodContext1.hashCode(), methodContext3.hashCode());
        assertNotEquals(methodContext1.hashCode(), methodContext4.hashCode());
        assertNotEquals(methodContext1.hashCode(), methodContext5.hashCode());
    }
}
