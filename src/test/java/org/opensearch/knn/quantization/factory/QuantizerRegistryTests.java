/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import org.junit.BeforeClass;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.quantizer.MultiBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.Quantizer;

public class QuantizerRegistryTests extends KNNTestCase {

    @BeforeClass
    public static void setup() {
        try {
            QuantizerRegistry.register(
                ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.ONE_BIT),
                new OneBitScalarQuantizer()
            );
            QuantizerRegistry.register(
                ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.TWO_BIT),
                new MultiBitScalarQuantizer(2)
            );
            QuantizerRegistry.register(
                ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.FOUR_BIT),
                new MultiBitScalarQuantizer(4)
            );
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("already registered"));
        }
    }

    public void testRegisterAndGetQuantizer() {
        // Test for OneBitScalarQuantizer
        ScalarQuantizationParams oneBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        Quantizer<Float[], Byte[]> oneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        assertEquals(oneBitQuantizer.getClass(), OneBitScalarQuantizer.class);

        // Test for MultiBitScalarQuantizer (2-bit)
        ScalarQuantizationParams twoBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        Quantizer<Float[], Byte[]> twoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        assertEquals(twoBitQuantizer.getClass(), MultiBitScalarQuantizer.class);

        // Test for MultiBitScalarQuantizer (4-bit)
        ScalarQuantizationParams fourBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.FOUR_BIT).build();
        Quantizer<Float[], Byte[]> fourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        assertEquals(fourBitQuantizer.getClass(), MultiBitScalarQuantizer.class);
    }

    public void testQuantizerRegistryIsSingleton() {
        // Ensure the same instance is returned for the same type identifier
        ScalarQuantizationParams oneBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        Quantizer<Float[], Byte[]> firstOneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        Quantizer<Float[], Byte[]> secondOneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        assertSame(firstOneBitQuantizer, secondOneBitQuantizer);

        // Ensure the same instance is returned for the same type identifier (2-bit)
        ScalarQuantizationParams twoBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        Quantizer<Float[], Byte[]> firstTwoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        Quantizer<Float[], Byte[]> secondTwoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        assertSame(firstTwoBitQuantizer, secondTwoBitQuantizer);

        // Ensure the same instance is returned for the same type identifier (4-bit)
        ScalarQuantizationParams fourBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.FOUR_BIT).build();
        Quantizer<Float[], Byte[]> firstFourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        Quantizer<Float[], Byte[]> secondFourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        assertSame(firstFourBitQuantizer, secondFourBitQuantizer);
    }

    public void testRegisterQuantizerThrowsExceptionWhenAlreadyRegistered() {
        ScalarQuantizationParams oneBitParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();

        // Attempt to register the same quantizer again should throw an exception
        assertThrows(IllegalArgumentException.class, () -> {
            QuantizerRegistry.register(
                ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.ONE_BIT),
                new OneBitScalarQuantizer()
            );
        });
    }
}
