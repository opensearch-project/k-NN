/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.quantizer.MultiBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.Quantizer;

public class QuantizerFactoryTests extends KNNTestCase {

    public void test_Lazy_Registration() {
        try {
            ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
            ScalarQuantizationParams paramsTwoBit = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
            ScalarQuantizationParams paramsFourBit = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
            Quantizer<Float[], Byte[]> oneBitQuantizer = QuantizerFactory.getQuantizer(params);
            Quantizer<Float[], Byte[]> quantizerTwoBit = QuantizerFactory.getQuantizer(paramsTwoBit);
            Quantizer<Float[], Byte[]> quantizerFourBit = QuantizerFactory.getQuantizer(paramsFourBit);
            assertEquals(OneBitScalarQuantizer.class, oneBitQuantizer.getClass());
            assertEquals(MultiBitScalarQuantizer.class, quantizerTwoBit.getClass());
            assertEquals(MultiBitScalarQuantizer.class, quantizerFourBit.getClass());
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("already registered"));
        }
    }

    public void testGetQuantizer_withNullParams() {
        try {
            QuantizerFactory.getQuantizer(null);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertEquals("Quantization parameters must not be null.", e.getMessage());
        }
    }
}
