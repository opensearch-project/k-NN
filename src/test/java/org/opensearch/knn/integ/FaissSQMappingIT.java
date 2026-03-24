/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;

import java.util.Map;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.PROPERTIES;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Integration tests for Faiss SQ encoder mapping validation.
 * Covers bits parameter requirements, type/bits conflicts, compression compatibility,
 * and various mapping combinations.
 */
public class FaissSQMappingIT extends KNNRestTestCase {

    private static final int TEST_DIMENSION = 128;
    private static final int NUM_DOCS = 10;
    private static final int K = 5;

    /**
     * Validates a bits=16 (fp16) index
     */
    private void validateFP16Index(String indexName, String mapping, Integer expectedBits, String expectedType, Boolean expectedClip)
        throws Exception {
        createKnnIndex(indexName, mapping);
        validateMapping(indexName, expectedBits, expectedType, expectedClip);
        addKNNDocs(indexName, FIELD_NAME, TEST_DIMENSION, 0, NUM_DOCS);
        validateKNNSearch(indexName, FIELD_NAME, TEST_DIMENSION, NUM_DOCS, K);
        deleteKNNIndex(indexName);
    }

    /**
     * Validates a bits=1 index at mapping level
     */
    private void validateMappingOnly(String indexName, String mapping, Integer expectedBits) throws Exception {
        createKnnIndex(indexName, mapping);
        validateMapping(indexName, expectedBits, null, null);
        deleteKNNIndex(indexName);
    }

    @SuppressWarnings("unchecked")
    private void validateMapping(String indexName, Integer expectedBits, String expectedType, Boolean expectedClip) throws Exception {
        Map<String, Object> actualMapping = getIndexMappingAsMap(indexName);
        assertNotNull(actualMapping);
        Map<String, Object> actualProps = (Map<String, Object>) actualMapping.get(PROPERTIES);
        assertNotNull("Mapping properties should not be null", actualProps);
        Map<String, Object> actualField = (Map<String, Object>) actualProps.get(FIELD_NAME);
        assertNotNull("Field [" + FIELD_NAME + "] should exist in mapping", actualField);
        assertEquals(TYPE_KNN_VECTOR, actualField.get(TYPE));
        assertEquals(TEST_DIMENSION, actualField.get(DIMENSION));

        Map<String, Object> method = (Map<String, Object>) actualField.get(KNN_METHOD);
        assertNotNull("Method should not be null", method);
        assertEquals(FAISS_NAME, method.get(KNN_ENGINE));
        Map<String, Object> params = (Map<String, Object>) method.get(PARAMETERS);
        assertNotNull("Method parameters should not be null", params);
        Map<String, Object> encoder = (Map<String, Object>) params.get(METHOD_ENCODER_PARAMETER);
        assertNotNull("Encoder should not be null", encoder);
        assertEquals(ENCODER_SQ, encoder.get(NAME));

        Map<String, Object> encoderParams = (Map<String, Object>) encoder.get(PARAMETERS);
        assertNotNull("Encoder parameters should not be null", encoderParams);
        if (expectedBits != null) {
            assertEquals(expectedBits, encoderParams.get(SQ_BITS));
        }
        if (expectedType != null) {
            assertEquals(expectedType, encoderParams.get(FAISS_SQ_TYPE));
        }
        if (expectedClip != null) {
            assertEquals(expectedClip, encoderParams.get(FAISS_SQ_CLIP));
        }
    }

    /**
     * Validates that index creation fails with a ResponseException containing the expected message.
     */
    private void validateMappingFails(String mapping, String expectedError) {
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
        assertTrue(
            "Expected error containing [" + expectedError + "] but got: " + ex.getMessage(),
            ex.getMessage().contains(expectedError)
        );
    }

    // --- bits required on 3.6.0+ ---

    public void testSQEncoder_whenNoBitsOnCurrentVersion_thenFail() throws Exception {
        String mapping = buildSQMapping(null, FAISS_SQ_ENCODER_FP16, null, null, null);
        validateMappingFails(mapping, "bits");
    }

    // --- bits=16 with type=fp16 (valid) ---

    public void testSQEncoder_whenBits16WithTypeFP16_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, null);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    public void testSQEncoder_whenBits16WithTypeFP16AndCompressionX2_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, CompressionLevel.x2.getName(), null, null);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    public void testSQEncoder_whenBits16WithTypeFP16AndOnDisk_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, Mode.ON_DISK.getName(), null);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    // --- bits=1 (type not allowed) ---

    public void testSQEncoder_whenBits1WithType_thenFail() throws Exception {
        String mapping = buildSQMapping(1, FAISS_SQ_ENCODER_FP16, null, null, null);
        validateMappingFails(mapping, "type");
    }

    public void testSQEncoder_whenBits1WithoutType_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, null, null, null);
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    public void testSQEncoder_whenBits1WithCompressionX32_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, CompressionLevel.x32.getName(), null, null);
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    // --- compression level mismatches ---

    public void testSQEncoder_whenBits1WithCompressionX2_thenFail() throws Exception {
        String mapping = buildSQMapping(1, null, CompressionLevel.x2.getName(), null, null);
        validateMappingFails(mapping, "incompatible");
    }

    public void testSQEncoder_whenBits16WithCompressionX32_thenFail() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, CompressionLevel.x32.getName(), null, null);
        validateMappingFails(mapping, "incompatible");
    }

    // --- invalid bits values ---

    public void testSQEncoder_whenBits2_thenFail() throws Exception {
        String mapping = buildSQMapping(2, null, null, null, null);
        validateMappingFails(mapping, "Unsupported bits value");
    }

    public void testSQEncoder_whenBits8_thenFail() throws Exception {
        String mapping = buildSQMapping(8, null, null, null, null);
        validateMappingFails(mapping, "Unsupported bits value");
    }

    // --- mode + encoder combinations ---

    public void testSQEncoder_whenBits16WithInMemoryMode_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, Mode.IN_MEMORY.getName(), null);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    // --- clip parameter: only valid with bits=16 (fp16) ---

    public void testSQEncoder_whenBits1WithClip_thenFail() throws Exception {
        String mapping = buildSQMapping(1, null, null, null, null, true);
        validateMappingFails(mapping, "clip");
    }

    public void testSQEncoder_whenBits16WithClip_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, null, true);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, true);
    }

    // --- IVF + sq(bits=1) not supported ---
    public void testSQEncoder_whenBits1WithIVF_thenFail() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 4)
            .field(METHOD_PARAMETER_NPROBES, 2)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(SQ_BITS, 1)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, builder.toString()));
        assertTrue(ex.getMessage().contains("Validation Failed"));
    }

    // --- Space type variations with bits=16 ---

    public void testSQEncoder_whenBits16WithL2_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, SpaceType.L2.getValue());
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    public void testSQEncoder_whenBits16WithIP_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, SpaceType.INNER_PRODUCT.getValue());
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    public void testSQEncoder_whenBits16WithCosine_thenSucceed() throws Exception {
        // Cosine rejects zero vectors from addKNNDocs, so validate mapping only
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, SpaceType.COSINESIMIL.getValue());
        validateMappingOnly(INDEX_NAME, mapping, 16);
    }

    // --- Space type variations with bits=1 ---

    public void testSQEncoder_whenBits1WithL2_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, null, null, SpaceType.L2.getValue());
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    public void testSQEncoder_whenBits1WithIP_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, null, null, SpaceType.INNER_PRODUCT.getValue());
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    public void testSQEncoder_whenBits1WithCosine_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, null, null, SpaceType.COSINESIMIL.getValue());
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    // --- bits=16 without type (should default to fp16) ---

    public void testSQEncoder_whenBits16WithoutType_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, null, null, null, null);
        validateFP16Index(INDEX_NAME, mapping, 16, null, null);
    }

    // --- bits=1 with mode variations ---

    public void testSQEncoder_whenBits1WithOnDiskMode_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, null, Mode.ON_DISK.getName(), null);
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    public void testSQEncoder_whenBits1WithInMemoryMode_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, null, Mode.IN_MEMORY.getName(), null);
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    // --- Full explicit combos ---

    public void testSQEncoder_whenBits16WithOnDiskAndCompressionX2_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, CompressionLevel.x2.getName(), Mode.ON_DISK.getName(), null);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    public void testSQEncoder_whenBits1WithOnDiskAndCompressionX32_thenSucceed() throws Exception {
        String mapping = buildSQMapping(1, null, CompressionLevel.x32.getName(), Mode.ON_DISK.getName(), null);
        validateMappingOnly(INDEX_NAME, mapping, 1);
    }

    // --- Data type: SQ only supports FLOAT ---

    public void testSQEncoder_whenBits1WithByteDataType_thenFail() throws Exception {
        String mapping = buildSQMappingWithDataType(1, null, VectorDataType.BYTE.getValue());
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits16WithByteDataType_thenFail() throws Exception {
        String mapping = buildSQMappingWithDataType(16, FAISS_SQ_ENCODER_FP16, VectorDataType.BYTE.getValue());
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits1WithBinaryDataType_thenFail() throws Exception {
        String mapping = buildSQMappingWithDataType(1, null, VectorDataType.BINARY.getValue());
        validateMappingFails(mapping, "Validation Failed");
    }

    // --- Additional compression level mismatches ---

    public void testSQEncoder_whenBits1WithCompressionX4_thenFail() throws Exception {
        String mapping = buildSQMapping(1, null, CompressionLevel.x4.getName(), null, null);
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits1WithCompressionX8_thenFail() throws Exception {
        String mapping = buildSQMapping(1, null, CompressionLevel.x8.getName(), null, null);
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits1WithCompressionX16_thenFail() throws Exception {
        String mapping = buildSQMapping(1, null, CompressionLevel.x16.getName(), null, null);
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits16WithCompressionX4_thenFail() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, CompressionLevel.x4.getName(), null, null);
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits16WithCompressionX8_thenFail() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, CompressionLevel.x8.getName(), null, null);
        validateMappingFails(mapping, "Validation Failed");
    }

    public void testSQEncoder_whenBits16WithCompressionX16_thenFail() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, CompressionLevel.x16.getName(), null, null);
        validateMappingFails(mapping, "Validation Failed");
    }

    // --- bits=16 only (no type, no clip) base case ---

    public void testSQEncoder_whenOnlyBits16_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, null, null, null, null);
        validateFP16Index(INDEX_NAME, mapping, 16, null, null);
    }

    // --- bits + type + clip combinations ---

    public void testSQEncoder_whenBits16WithTypeFP16AndClipTrue_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, null, true);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, true);
    }

    public void testSQEncoder_whenBits16WithTypeFP16AndClipFalse_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, FAISS_SQ_ENCODER_FP16, null, null, null, false);
        validateFP16Index(INDEX_NAME, mapping, 16, FAISS_SQ_ENCODER_FP16, null);
    }

    public void testSQEncoder_whenBits16WithNoTypeAndClipTrue_thenSucceed() throws Exception {
        String mapping = buildSQMapping(16, null, null, null, null, true);
        validateFP16Index(INDEX_NAME, mapping, 16, null, true);
    }

    public void testSQEncoder_whenBits1WithClipFalse_thenFail() throws Exception {
        // clip parameter is not applicable for bits=1, even when false
        String mapping = buildSQMapping(1, null, null, null, null, false);
        validateMappingFails(mapping, "clip");
    }

    private String buildSQMappingWithDataType(Integer bits, String type, String dataType) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, dataType)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS);
        if (bits != null) {
            builder.field(SQ_BITS, bits);
        }
        if (type != null) {
            builder.field(FAISS_SQ_TYPE, type);
        }
        builder.endObject().endObject().endObject().endObject().endObject().endObject().endObject();
        return builder.toString();
    }

    private String buildSQMapping(Integer bits, String type, String compressionLevel, String mode, String spaceType) throws Exception {
        return buildSQMapping(bits, type, compressionLevel, mode, spaceType, null);
    }

    private String buildSQMapping(Integer bits, String type, String compressionLevel, String mode, String spaceType, Boolean clip)
        throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION);

        if (mode != null) {
            builder.field(MODE_PARAMETER, mode);
        }
        if (compressionLevel != null) {
            builder.field(COMPRESSION_LEVEL_PARAMETER, compressionLevel);
        }

        builder.startObject(KNN_METHOD).field(NAME, METHOD_HNSW).field(KNN_ENGINE, FAISS_NAME);
        if (spaceType != null) {
            builder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType);
        }
        builder.startObject(PARAMETERS).startObject(METHOD_ENCODER_PARAMETER).field(NAME, ENCODER_SQ).startObject(PARAMETERS);

        if (bits != null) {
            builder.field(SQ_BITS, bits);
        }
        if (type != null) {
            builder.field(FAISS_SQ_TYPE, type);
        }
        if (clip != null) {
            builder.field(FAISS_SQ_CLIP, clip);
        }

        builder.endObject().endObject().endObject().endObject().endObject().endObject().endObject();

        return builder.toString();
    }
}
