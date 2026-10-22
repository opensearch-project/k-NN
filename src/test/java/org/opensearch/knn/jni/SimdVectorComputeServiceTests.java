/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.junit.Test;
import org.opensearch.knn.KNNTestCase;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.jni.SimdVectorComputeService.SimilarityFunctionType;

public class SimdVectorComputeServiceTests extends KNNTestCase {
    /**
     * The numbers the native side dispatches on (NativeSimilarityFunctionType in similarity_function.h). Changing one of
     * these is a JNI wire contract break, so they are spelled out here rather than derived from the enum.
     */
    private static final Map<SimilarityFunctionType, Integer> PINNED_WIRE_VALUES = Map.of(
        SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT,
        0,
        SimilarityFunctionType.FP16_L2,
        1,
        SimilarityFunctionType.SQ_IP,
        2,
        SimilarityFunctionType.SQ_L2,
        3,
        SimilarityFunctionType.BF16_MAXIMUM_INNER_PRODUCT,
        4,
        SimilarityFunctionType.BF16_L2,
        5,
        SimilarityFunctionType.FP16_COSINE,
        6,
        SimilarityFunctionType.SQ_COSINE,
        7
    );

    @Test
    public void testWireValuesArePinned() {
        assertEquals(
            "Every SimilarityFunctionType must have a pinned wire value; add the new constant to PINNED_WIRE_VALUES",
            SimilarityFunctionType.values().length,
            PINNED_WIRE_VALUES.size()
        );
        for (final SimilarityFunctionType type : SimilarityFunctionType.values()) {
            assertEquals("Wire value of " + type + " changed", (int) PINNED_WIRE_VALUES.get(type), type.getWireValue());
        }
    }

    @Test
    public void testWireValuesAreUnique() {
        final Set<Integer> seen = new HashSet<>();
        for (final SimilarityFunctionType type : SimilarityFunctionType.values()) {
            assertTrue("Duplicate wire value " + type.getWireValue() + " for " + type, seen.add(type.getWireValue()));
        }
    }
}
