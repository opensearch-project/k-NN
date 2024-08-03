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
        assertFalse(isRegisteredFieldAccessible());
        Quantizer<?, ?> quantizer = QuantizerFactory.getQuantizer(params);
        assertTrue(quantizer instanceof OneBitScalarQuantizer);
        assertTrue(isRegisteredFieldAccessible());
    }

    public void testGetQuantizer_withOneBitSQParams() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        Quantizer<?, ?> quantizer = QuantizerFactory.getQuantizer(params);
        assertTrue(quantizer instanceof OneBitScalarQuantizer);
    }

    public void testGetQuantizer_withTwoBitSQParams() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        Quantizer<?, ?> quantizer = QuantizerFactory.getQuantizer(params);
        assertTrue(quantizer instanceof MultiBitScalarQuantizer);
    }

    public void testGetQuantizer_withFourBitSQParams() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        Quantizer<?, ?> quantizer = QuantizerFactory.getQuantizer(params);
        assertTrue(quantizer instanceof MultiBitScalarQuantizer);
    }

    public void testGetQuantizer_withNullParams() {
        try {
            QuantizerFactory.getQuantizer(null);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertEquals("Quantization parameters must not be null.", e.getMessage());
        }
    }

    public void testConcurrentRegistration() throws InterruptedException {
        Runnable task = () -> {
            ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
            QuantizerFactory.getQuantizer(params);
        };

        Thread thread1 = new Thread(task);
        Thread thread2 = new Thread(task);
        thread1.start();
        thread2.start();
        thread1.join();
        thread2.join();
        assertTrue(isRegisteredFieldAccessible());
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
