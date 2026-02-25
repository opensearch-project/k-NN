/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import com.google.common.collect.ImmutableMap;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import org.opensearch.core.common.io.stream.StreamInput;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
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

        assertEquals(KNNEngine.UNDEFINED, knnMethodContext.getKnnEngine());
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

    public void testValidateVectorDataType_whenBinaryFaissHNSW_thenValid() {
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_HNSW, VectorDataType.BINARY, SpaceType.HAMMING, null);
    }

    public void testValidateVectorDataType_whenBinaryNonFaiss_thenException() {
        validateValidateVectorDataType(
            KNNEngine.NMSLIB,
            KNNConstants.METHOD_HNSW,
            VectorDataType.BINARY,
            SpaceType.HAMMING,
            "UnsupportedMethod"
        );
    }

    public void testValidateVectorDataType_whenByte_thenValid() {
        validateValidateVectorDataType(KNNEngine.LUCENE, KNNConstants.METHOD_HNSW, VectorDataType.BYTE, SpaceType.L2, null);
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_HNSW, VectorDataType.BYTE, SpaceType.L2, null);
    }

    public void testValidateVectorDataType_whenByte_thenException() {
        validateValidateVectorDataType(KNNEngine.NMSLIB, KNNConstants.METHOD_IVF, VectorDataType.BYTE, SpaceType.L2, "UnsupportedMethod");
    }

    public void testValidateVectorDataType_whenFloat_thenValid() {
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_HNSW, VectorDataType.FLOAT, SpaceType.L2, null);
        validateValidateVectorDataType(KNNEngine.LUCENE, KNNConstants.METHOD_HNSW, VectorDataType.FLOAT, SpaceType.L2, null);
        validateValidateVectorDataType(KNNEngine.NMSLIB, KNNConstants.METHOD_HNSW, VectorDataType.FLOAT, SpaceType.L2, null);
    }

    public void testWriteTo_withNullParameters() throws IOException {
        MethodComponentContext methodComponent = new MethodComponentContext("test-method", null);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        methodComponent.writeTo(streamOutput);

        MethodComponentContext deserialized = new MethodComponentContext(streamOutput.bytes().streamInput());

        // Ensure parameters are empty (not null) for safety
        assertNotNull(deserialized.getParameters());
        assertTrue(deserialized.getParameters().isEmpty());
    }

    public void testWriteToReadFrom_withValidParameters() throws IOException {
        Map<String, Object> parameters = ImmutableMap.of("param1", 10, "param2", "value");
        MethodComponentContext original = new MethodComponentContext("test-method", parameters);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        original.writeTo(streamOutput);

        MethodComponentContext deserialized = new MethodComponentContext(streamOutput.bytes().streamInput());

        assertEquals(original, deserialized);
        assertEquals(2, deserialized.getParameters().size());
        assertEquals(10, deserialized.getParameters().get("param1"));
        assertEquals("value", deserialized.getParameters().get("param2"));
    }

    public void testBackwardCompatibility_readFromOldVersion() throws IOException {
        // Simulating an older version that did not write parameters explicitly
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        streamOutput.writeString("old-method");
        StreamInput streamInput = streamOutput.bytes().streamInput();
        streamInput.setVersion(Version.V_2_19_0);

        MethodComponentContext deserialized = new MethodComponentContext(streamInput);

        // Ensure parameters are still handled gracefully
        assertNotNull(deserialized.getParameters());
        assertTrue(deserialized.getParameters().isEmpty());
    }

    public void testReadFrom_beforeVersion3_0_0() throws IOException {
        // Simulate a stream written from a version before 3.0.0 (parameters not explicitly stored)
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        streamOutput.writeString("test-method");
        StreamInput streamInput = streamOutput.bytes().streamInput();
        streamInput.setVersion(Version.V_2_19_0);

        MethodComponentContext deserialized = new MethodComponentContext(streamInput);
        // Ensure parameters default to an empty map for older versions
        assertNotNull(deserialized.getParameters());
        assertTrue(deserialized.getParameters().isEmpty());
    }

    private void validateValidateVectorDataType(
        final KNNEngine knnEngine,
        final String methodName,
        final VectorDataType vectorDataType,
        final SpaceType spaceType,
        final String expectedErrMsg
    ) {
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodName, Collections.emptyMap());
        KNNMethodContext methodContext = new KNNMethodContext(knnEngine, spaceType, methodComponentContext);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(vectorDataType)
            .dimension(8)
            .versionCreated(Version.CURRENT)
            .build();
        if (expectedErrMsg == null) {
            assertNull(knnEngine.validateMethod(methodContext, knnMethodConfigContext));
        } else {
            assertNotNull(knnEngine.validateMethod(methodContext, knnMethodConfigContext));
        }
    }

}
