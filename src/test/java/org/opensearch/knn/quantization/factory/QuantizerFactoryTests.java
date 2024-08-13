/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import org.junit.Before;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.quantizer.MultiBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;

public class QuantizerFactoryTests extends KNNTestCase {

    @Before
    public void resetIsRegisteredFlag() throws NoSuchFieldException, IllegalAccessException {
        Field isRegisteredField = QuantizerFactory.class.getDeclaredField("isRegistered");
        isRegisteredField.setAccessible(true);
        AtomicBoolean isRegistered = (AtomicBoolean) isRegisteredField.get(null);
        isRegistered.set(false);
    }

    public void test_Lazy_Registration() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        ScalarQuantizationParams paramsTwoBit = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        ScalarQuantizationParams paramsFourBit = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        assertFalse(isRegisteredFieldAccessible());
        Quantizer<?, ?> quantizer = QuantizerFactory.getQuantizer(params);
        Quantizer<?, ?> quantizerTwoBit = QuantizerFactory.getQuantizer(paramsTwoBit);
        Quantizer<?, ?> quantizerFourBit = QuantizerFactory.getQuantizer(paramsFourBit);
        assertTrue(quantizerFourBit instanceof MultiBitScalarQuantizer);
        assertTrue(quantizerTwoBit instanceof MultiBitScalarQuantizer);
        assertTrue(quantizer instanceof OneBitScalarQuantizer);
        assertTrue(isRegisteredFieldAccessible());
    }

    public void testGetQuantizer_withNullParams() {
        try {
            QuantizerFactory.getQuantizer(null);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertEquals("Quantization parameters must not be null.", e.getMessage());
        }
    }

    private boolean isRegisteredFieldAccessible() {
        try {
            Field isRegisteredField = QuantizerFactory.class.getDeclaredField("isRegistered");
            isRegisteredField.setAccessible(true);
            AtomicBoolean isRegistered = (AtomicBoolean) isRegisteredField.get(null);
            return isRegistered.get();
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access isRegistered field.");
            return false;
        }
    }
}
