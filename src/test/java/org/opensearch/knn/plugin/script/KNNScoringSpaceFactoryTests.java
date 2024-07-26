/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.index.mapper.NumberFieldMapper;

import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNScoringSpaceFactoryTests extends KNNTestCase {
    public void testValidSpaces() {
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldType.getDimension()).thenReturn(3);
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldTypeBinary = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldTypeBinary.getDimension()).thenReturn(24);
        when(knnVectorFieldTypeBinary.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        NumberFieldMapper.NumberFieldType numberFieldType = new NumberFieldMapper.NumberFieldType(
            "field",
            NumberFieldMapper.NumberType.LONG
        );
        List<Float> floatQueryObject = List.of(1.0f, 1.0f, 1.0f);
        Long longQueryObject = 0L;

        assertTrue(
            KNNScoringSpaceFactory.create(SpaceType.L2.getValue(), floatQueryObject, knnVectorFieldType) instanceof KNNScoringSpace.L2
        );
        assertTrue(
            KNNScoringSpaceFactory.create(
                SpaceType.COSINESIMIL.getValue(),
                floatQueryObject,
                knnVectorFieldType
            ) instanceof KNNScoringSpace.CosineSimilarity
        );
        assertTrue(
            KNNScoringSpaceFactory.create(
                SpaceType.INNER_PRODUCT.getValue(),
                floatQueryObject,
                knnVectorFieldType
            ) instanceof KNNScoringSpace.InnerProd
        );
        assertTrue(
            KNNScoringSpaceFactory.create(
                SpaceType.HAMMING.getValue(),
                floatQueryObject,
                knnVectorFieldTypeBinary
            ) instanceof KNNScoringSpace.Hamming
        );
        assertTrue(
            KNNScoringSpaceFactory.create(
                KNNScoringSpaceFactory.HAMMING_BIT,
                longQueryObject,
                numberFieldType
            ) instanceof KNNScoringSpace.HammingBit
        );
    }

    public void testInvalidSpace() {
        List<Float> floatQueryObject = List.of(1.0f, 1.0f, 1.0f);
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldType.getDimension()).thenReturn(3);
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldTypeBinary = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldTypeBinary.getDimension()).thenReturn(24);
        when(knnVectorFieldTypeBinary.getVectorDataType()).thenReturn(VectorDataType.BINARY);

        // Verify
        expectThrows(IllegalArgumentException.class, () -> KNNScoringSpaceFactory.create(SpaceType.L2.getValue(), null, null));
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNScoringSpaceFactory.create(SpaceType.L2.getValue(), floatQueryObject, knnVectorFieldTypeBinary)
        );
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNScoringSpaceFactory.create(SpaceType.HAMMING.getValue(), floatQueryObject, knnVectorFieldType)
        );
    }
}
