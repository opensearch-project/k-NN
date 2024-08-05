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

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.KNNTestCase;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

public class SpaceTypeTests extends KNNTestCase {

    public void testGetVectorSimilarityFunction_l2() {
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction());
    }

    public void testGetVectorSimilarityFunction_invalid() {
        expectThrows(UnsupportedOperationException.class, SpaceType.L1::getKnnVectorSimilarityFunction);
    }

    public void testValidateVectorDataType_whenCalled_thenReturn() {
        Map<SpaceType, Set<VectorDataType>> expected = Map.of(
            SpaceType.UNDEFINED,
            Collections.emptySet(),
            SpaceType.L2,
            Set.of(VectorDataType.FLOAT, VectorDataType.BYTE),
            SpaceType.COSINESIMIL,
            Set.of(VectorDataType.FLOAT, VectorDataType.BYTE),
            SpaceType.L1,
            Set.of(VectorDataType.FLOAT, VectorDataType.BYTE),
            SpaceType.LINF,
            Set.of(VectorDataType.FLOAT, VectorDataType.BYTE),
            SpaceType.INNER_PRODUCT,
            Set.of(VectorDataType.FLOAT, VectorDataType.BYTE),
            SpaceType.HAMMING,
            Set.of(VectorDataType.BINARY)
        );

        for (SpaceType spaceType : SpaceType.values()) {
            for (VectorDataType vectorDataType : VectorDataType.values()) {
                if (expected.get(spaceType).isEmpty()) {
                    Exception ex = expectThrows(IllegalStateException.class, () -> spaceType.validateVectorDataType(vectorDataType));
                    assertTrue(ex.getMessage().contains("Unsupported method"));
                    continue;
                }

                if (expected.get(spaceType).contains(vectorDataType)) {
                    spaceType.validateVectorDataType(vectorDataType);
                } else {
                    Exception ex = expectThrows(IllegalArgumentException.class, () -> spaceType.validateVectorDataType(vectorDataType));
                    assertTrue(ex.getMessage().contains("is not supported"));
                }
            }
        }
    }
}
