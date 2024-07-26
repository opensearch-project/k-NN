/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

import static org.hamcrest.Matchers.containsString;

public class KNNValidationUtilTests extends KNNTestCase {
    public void testValidateVectorDimension_whenBinary_thenVectorSizeShouldBeEightTimesLarger() {
        int vectorLength = randomInt(100) + 1;
        Exception ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNValidationUtil.validateVectorDimension(vectorLength, vectorLength, VectorDataType.BINARY)
        );
        assertThat(
            ex.getMessage(),
            containsString("The dimension of the binary vector must be 8 times the length of the provided vector.")
        );

        // Expect no exception
        KNNValidationUtil.validateVectorDimension(vectorLength * Byte.SIZE, vectorLength, VectorDataType.BINARY);
    }

    public void testValidateVectorDimension_whenNonBinary_thenVectorSizeShouldBeSameAsDimension() {
        int dimension = randomInt(100);
        VectorDataType vectorDataType = randomInt(1) == 0 ? VectorDataType.FLOAT : VectorDataType.BYTE;
        Exception ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNValidationUtil.validateVectorDimension(dimension, dimension + 1, vectorDataType)
        );
        assertThat(ex.getMessage(), containsString("Vector dimension mismatch"));

        // Expect no exception
        KNNValidationUtil.validateVectorDimension(dimension, dimension, vectorDataType);
    }
}
