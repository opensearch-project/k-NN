/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import java.util.Map;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.PROPERTIES;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

/**
 * Integration tests for Lucene SQ encoder mapping validation.
 * Covers bits parameter requirements, type/bits conflicts, compression compatibility,
 * and various mapping combinations.
 */
public class LuceneSQMappingIT extends KNNRestTestCase {
    private static final int TEST_DIMENSION = 128;
    private static final int NUM_DOCS = 10;
    private static final String TYPE_NESTED_FIELD = "nested";
    private static final String NESTED_FIELD_NAME = "my_nested_field";

    private void validateIndex(
        String indexName,
        String mapping,
        String nestedFieldName,
        String vectorFieldName,
        String encoderName,
        Integer bits
    ) throws Exception {
        String fieldPath = nestedFieldName != null ? nestedFieldName + "." + vectorFieldName : vectorFieldName;
        createKnnIndex(indexName, mapping);
        validateMapping(nestedFieldName, indexName, bits, encoderName);
        if (nestedFieldName != null) {
            bulkIngestRandomVectorsWithNestedField(indexName, fieldPath, NUM_DOCS, TEST_DIMENSION);
        } else {
            addKNNDocs(indexName, vectorFieldName, TEST_DIMENSION, 0, NUM_DOCS);
        }
        deleteKNNIndex(indexName);
    }

    private void validateMappingFails(String mapping, String expectedError) {
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
        assertTrue(
            "Expected error containing [" + expectedError + "] but got: " + ex.getMessage(),
            ex.getMessage().contains(expectedError)
        );
    }

    private String buildSQMapping(Integer bits, String compressionLevel, String mode) throws Exception {
        return buildSQMapping(bits, compressionLevel, mode, false);
    }

    private String buildSQMapping(Integer bits, String compressionLevel, String mode, boolean nested) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(PROPERTIES);

        if (nested) {
            builder.startObject(NESTED_FIELD_NAME).field(TYPE, TYPE_NESTED_FIELD).startObject(PROPERTIES);
        }

        builder.startObject(FIELD_NAME).field(TYPE, TYPE_KNN_VECTOR).field(DIMENSION, TEST_DIMENSION);

        if (mode != null) {
            builder.field(MODE_PARAMETER, mode);
        }
        if (compressionLevel != null) {
            builder.field(COMPRESSION_LEVEL_PARAMETER, compressionLevel);
        }

        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ);
        if (bits != null) {
            builder.startObject(PARAMETERS).field(SQ_BITS, bits).endObject();
        }

        if (nested) {
            builder.endObject().endObject();
        }

        builder.endObject().endObject().endObject().endObject().endObject().endObject();
        return builder.toString();
    }

    @SuppressWarnings("unchecked")
    private void validateMapping(String nestedFieldName, String indexName, Integer expectedBits, String expectedEncoderName)
        throws Exception {
        Map<String, Object> actualMapping = getIndexMappingAsMap(indexName);
        assertNotNull(actualMapping);
        Map<String, Object> props = (Map<String, Object>) actualMapping.get(PROPERTIES);
        assertNotNull("Mapping properties should not be null", props);

        Map<String, Object> fieldProps = props;
        if (nestedFieldName != null) {
            Map<String, Object> nestedField = (Map<String, Object>) props.get(nestedFieldName);
            assertNotNull("Nested field [" + nestedFieldName + "] should exist in mapping", nestedField);
            fieldProps = (Map<String, Object>) nestedField.get(PROPERTIES);
            assertNotNull("Nested field properties should not be null", fieldProps);
        }

        Map<String, Object> actualField = (Map<String, Object>) fieldProps.get(FIELD_NAME);
        assertNotNull("Field [" + FIELD_NAME + "] should exist in mapping", actualField);
        assertEquals(TYPE_KNN_VECTOR, actualField.get(TYPE));
        assertEquals(TEST_DIMENSION, actualField.get(DIMENSION));

        if (expectedEncoderName != null) {
            Map<String, Object> method = (Map<String, Object>) actualField.get(KNN_METHOD);
            assertNotNull("Method should not be null", method);
            assertEquals(LUCENE_NAME, method.get(KNN_ENGINE));

            Map<String, Object> params = (Map<String, Object>) method.get(PARAMETERS);
            assertNotNull("Method parameters should not be null", params);
            Map<String, Object> encoder = (Map<String, Object>) params.get(METHOD_ENCODER_PARAMETER);
            assertNotNull("Encoder should not be null", encoder);
            assertEquals(expectedEncoderName, encoder.get(NAME));

            if (expectedBits != null) {
                Map<String, Object> encoderParams = (Map<String, Object>) encoder.get(PARAMETERS);
                assertNotNull("Encoder parameters should not be null", encoderParams);
                assertEquals(expectedBits, encoderParams.get(SQ_BITS));
            }
        }
    }

    private String buildCompressionOnlyMapping(String compressionLevel, String mode) throws Exception {
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

        builder.endObject().endObject().endObject();
        return builder.toString();
    }

    // --- Compression level mismatch tests ---

    // 4x compression with bits = 1
    public void testMapping_whenBits1WithX4Compression_thenFail() throws Exception {
        validateMappingFails(buildSQMapping(1, CompressionLevel.x4.getName(), null), "incompatible");
    }

    // 32x compression with bits = 7
    public void testMapping_whenBits7WithX32Compression_thenFail() throws Exception {
        validateMappingFails(buildSQMapping(7, CompressionLevel.x32.getName(), null), "incompatible");
    }

    // --- Valid base test cases ---

    // Only compression 32x specified
    public void testMapping_whenX32CompressionOnly_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildCompressionOnlyMapping(CompressionLevel.x32.getName(), null), null, FIELD_NAME, null, null);
    }

    // Only compression 4x specified
    public void testMapping_whenX4CompressionOnly_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildCompressionOnlyMapping(CompressionLevel.x4.getName(), null), null, FIELD_NAME, null, null);
    }

    // Bits = 1, no compression specified
    public void testMapping_whenBits1NoCompression_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(1, null, null), null, FIELD_NAME, ENCODER_SQ, 1);
    }

    // Bits = 7, no compression specified
    public void testMapping_whenBits7NoCompression_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(7, null, null), null, FIELD_NAME, ENCODER_SQ, 7);
    }

    // Bits = 1, 32x compression specified
    public void testMapping_whenBits1WithX32Compression_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(1, CompressionLevel.x32.getName(), null), null, FIELD_NAME, ENCODER_SQ, 1);
    }

    // Bits = 7, 4x compression specified
    public void testMapping_whenBits7WithX4Compression_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(7, CompressionLevel.x4.getName(), null), null, FIELD_NAME, ENCODER_SQ, 7);
    }

    // --- Valid cases when Encoder Specified ---

    // Encoder is specified as SQ with mode: in_memory
    public void testMapping_whenEncoderSQWithInMemoryMode_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(7, null, Mode.IN_MEMORY.getName()), null, FIELD_NAME, ENCODER_SQ, 7);
    }

    // Encoder is specified as SQ with mode: on_disk
    public void testMapping_whenEncoderSQWithOnDiskMode_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(1, null, Mode.ON_DISK.getName()), null, FIELD_NAME, ENCODER_SQ, 1);
    }

    // --- Invalid bits provided ---

    // Bits = 3
    public void testMapping_whenBits3_thenFail() throws Exception {
        validateMappingFails(buildSQMapping(3, null, null), "bits");
    }

    // Bits = -1
    public void testMapping_whenBitsNegative1_thenFail() throws Exception {
        validateMappingFails(buildSQMapping(-1, null, null), "bits");
    }

    // SQ encoder specified and no bits is provided in current version - should fail
    public void testMapping_whenEncoderSQNoBitsCurrentVersion_thenFail() throws Exception {
        validateMappingFails(buildSQMapping(null, null, null), "required");
    }

    // --- Nested field valid cases ---

    // Nested field with SQ Encoder Specified and Bits = 1
    public void testMapping_whenNestedFieldWithBits1_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(1, null, null, true), NESTED_FIELD_NAME, FIELD_NAME, ENCODER_SQ, 1);
    }

    // Nested field with on_disk specified
    public void testMapping_whenNestedFieldWithOnDisk_thenSuccess() throws Exception {
        validateIndex(INDEX_NAME, buildSQMapping(1, null, Mode.ON_DISK.getName(), true), NESTED_FIELD_NAME, FIELD_NAME, ENCODER_SQ, 1);
    }
}
