/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.clipVectorValueToFP16Range;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.validateFP16VectorValue;

public class FaissFP16UtilTests extends KNNTestCase {

    public void testValidateFp16VectorValue_outOfRange_throwsException() {
        IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> validateFP16VectorValue(65505.25f));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        IllegalArgumentException ex1 = expectThrows(IllegalArgumentException.class, () -> validateFP16VectorValue(-65525.65f));
        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );
    }

    public void testClipVectorValuetoFP16Range_succeed() {
        assertEquals(65504.0f, clipVectorValueToFP16Range(65504.10f), 0.0f);
        assertEquals(65504.0f, clipVectorValueToFP16Range(1000000.89f), 0.0f);
        assertEquals(-65504.0f, clipVectorValueToFP16Range(-65504.10f), 0.0f);
        assertEquals(-65504.0f, clipVectorValueToFP16Range(-1000000.89f), 0.0f);
    }

}
