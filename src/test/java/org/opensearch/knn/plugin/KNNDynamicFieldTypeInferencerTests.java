/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KNNDynamicFieldTypeInferencerTests extends KNNTestCase {

    private final KNNDynamicFieldTypeInferencer inferencer = new KNNDynamicFieldTypeInferencer();

    /** A supplier over a flat numeric array of the given length. */
    private FieldValueParserSupplier numericArray(int length) {
        String json = "[" + IntStream.range(0, length).mapToObj(i -> "0.1").collect(Collectors.joining(",")) + "]";
        return supplierOver(json);
    }

    private FieldValueParserSupplier supplierOver(String json) {
        return new FieldValueParserSupplier(
            MediaTypeRegistry.JSON,
            LoggingDeprecationHandler.INSTANCE,
            json.getBytes(StandardCharsets.UTF_8)
        );
    }

    /**
     * The inferencer must declare exactly knn_vector as its supported type. Core validates at startup
     * that this set is non-empty and that every type is registered by the plugin via getMappers(), and
     * enforces at parse time that a claim's type is in this set — so this contract is load-bearing.
     */
    public void testSupportedTypesIsKnnVector() {
        assertEquals(Set.of(KNNVectorFieldMapper.CONTENT_TYPE), inferencer.supportedTypes());
    }

    public void testClaimsArrayAtThreshold() throws IOException {
        Map<String, Object> config = inferencer.inferFieldType(numericArray(128));
        assertNotNull(config);
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(128, config.get("dimension"));
    }

    /**
     * Every type the inferencer actually produces must be within its declared supportedTypes(): this is
     * exactly the invariant core enforces at parse time, so a claim outside the declared set would be
     * silently dropped. Sweep the claiming range and assert consistency.
     */
    public void testEveryClaimedTypeIsWithinSupportedTypes() throws IOException {
        Set<String> supported = inferencer.supportedTypes();
        for (int dim : new int[] { 128, 256, 384, 512, 768, 1024 }) {
            Map<String, Object> config = inferencer.inferFieldType(numericArray(dim));
            assertNotNull("dim " + dim + " should be claimed", config);
            assertTrue(
                "claimed type [" + config.get("type") + "] must be within supportedTypes() " + supported,
                supported.contains(config.get("type"))
            );
        }
    }

    public void testClaimsMultipleOfEightAboveThreshold() throws IOException {
        // 256 is >= 128 and a multiple of 8 — claimed.
        Map<String, Object> config = inferencer.inferFieldType(numericArray(256));
        assertNotNull("256-dim array must be inferred as knn_vector", config);
        assertEquals(256, config.get("dimension"));
    }

    public void testNonMultipleOfEightNotClaimed() throws IOException {
        // 130 and 300 are >= 128 but not multiples of 8 — must NOT be claimed (%8 gate).
        assertNull("130-dim array is not a multiple of 8", inferencer.inferFieldType(numericArray(130)));
        assertNull("300-dim array is not a multiple of 8", inferencer.inferFieldType(numericArray(300)));
    }

    public void testBelowThresholdNotClaimed() throws IOException {
        // 128 is a multiple of 8 but the check is dimension first; 120 is a multiple of 8 but below threshold.
        assertNull(inferencer.inferFieldType(numericArray(120)));
    }

    public void testNonNumericArrayNotClaimed() throws IOException {
        // 128 elements but one is a string → not a flat numeric array.
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 127; i++) {
            sb.append("0.1,");
        }
        sb.append("\"x\"]");
        assertNull(inferencer.inferFieldType(supplierOver(sb.toString())));
    }

    public void testNonArrayNotClaimed() throws IOException {
        assertNull(inferencer.inferFieldType(supplierOver("\"not-an-array\"")));
    }

    /**
     * Loop test: sweep a wide range of dimensions and assert the gate exactly — claimed iff the length
     * is >= MIN_VECTOR_DIMENSION AND a multiple of 8. Covers the boundary (127/128), the multiple-of-8
     * boundary, and large real embedding sizes.
     */
    public void testDimensionSweepMatchesGateExactly() throws IOException {
        int min = KNNDynamicFieldTypeInferencer.MIN_VECTOR_DIMENSION; // 128
        for (int dim = 1; dim <= 1024; dim++) {
            Map<String, Object> config = inferencer.inferFieldType(numericArray(dim));
            boolean expectedClaim = dim >= min && dim % 8 == 0;
            if (expectedClaim) {
                assertNotNull("dim " + dim + " (>=128 and multiple of 8) must be claimed", config);
                assertEquals("type for dim " + dim, KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
                assertEquals("dimension for dim " + dim, dim, config.get("dimension"));
            } else {
                assertNull("dim " + dim + " must NOT be claimed (below 128 or not a multiple of 8)", config);
            }
        }
    }

    /** Loop test over the exact multiples of 8 at/above threshold that real models use. */
    public void testCommonEmbeddingDimensionsClaimed() throws IOException {
        for (int dim : new int[] { 128, 256, 384, 512, 768, 1024 }) {
            Map<String, Object> config = inferencer.inferFieldType(numericArray(dim));
            assertNotNull("common embedding dim " + dim + " must be claimed", config);
            assertEquals(dim, config.get("dimension"));
        }
    }

    /** Loop test: near-threshold non-multiples-of-8 just above 128 are all declined. */
    public void testNearThresholdNonMultiplesDeclined() throws IOException {
        for (int dim : new int[] { 129, 130, 131, 132, 133, 134, 135 }) {
            assertNull("dim " + dim + " (>=128 but not a multiple of 8) must be declined", inferencer.inferFieldType(numericArray(dim)));
        }
    }

    /** Integer-valued arrays are still numeric (VALUE_NUMBER), so they are claimed like float arrays. */
    public void testIntegerArrayClaimed() throws IOException {
        String json = "[" + IntStream.range(0, 128).mapToObj(i -> "1").collect(Collectors.joining(",")) + "]";
        Map<String, Object> config = inferencer.inferFieldType(supplierOver(json));
        assertNotNull("integer-valued array must be claimed (integers are VALUE_NUMBER)", config);
        assertEquals(128, config.get("dimension"));
    }

    /** Empty array is not claimed (length 0 < threshold). */
    public void testEmptyArrayNotClaimed() throws IOException {
        assertNull(inferencer.inferFieldType(supplierOver("[]")));
    }

    /** Nested array element disqualifies (START_ARRAY is not VALUE_NUMBER). */
    public void testNestedArrayNotClaimed() throws IOException {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 127; i++) {
            sb.append("0.1,");
        }
        sb.append("[1,2]]"); // 128th element is a nested array
        assertNull(inferencer.inferFieldType(supplierOver(sb.toString())));
    }
}
