/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.List;
import java.util.Optional;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNScoringSpaceFactoryTests extends KNNTestCase {
    public void testValidSpaces() {
        KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
        when(knnVectorFieldType.getKnnMethodConfigContext()).thenReturn(
            Optional.ofNullable(
                getKnnVectorFieldTypeConfigSupplierForMethodType(getDefaultKNNMethodContext(), 3).get().getKnnMethodConfigContext()
            )
        );
        KNNVectorFieldType knnVectorFieldTypeBinary = mock(KNNVectorFieldType.class);
        when(knnVectorFieldTypeBinary.getKnnMethodConfigContext()).thenReturn(
            Optional.ofNullable(
                getKnnVectorFieldTypeConfigSupplierForMethodType(getDefaultBinaryKNNMethodContext(), 24).get().getKnnMethodConfigContext()
            )
        );
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
        KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
        when(knnVectorFieldType.getKnnMethodConfigContext()).thenReturn(
            Optional.ofNullable(
                getKnnVectorFieldTypeConfigSupplierForMethodType(getDefaultKNNMethodContext(), 3).get().getKnnMethodConfigContext()
            )
        );
        KNNVectorFieldType knnVectorFieldTypeBinary = mock(KNNVectorFieldType.class);
        when(knnVectorFieldTypeBinary.getKnnMethodConfigContext()).thenReturn(
            Optional.ofNullable(
                getKnnVectorFieldTypeConfigSupplierForMethodType(getDefaultBinaryKNNMethodContext(), 24).get().getKnnMethodConfigContext()
            )
        );
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
