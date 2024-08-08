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
    }

    public void testRegisterAndGetQuantizer() {
        // Test for OneBitScalarQuantizer
        ScalarQuantizationParams oneBitParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        Quantizer<?, ?> oneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        assertTrue(oneBitQuantizer instanceof OneBitScalarQuantizer);

        // Test for MultiBitScalarQuantizer (2-bit)
        ScalarQuantizationParams twoBitParams = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        Quantizer<?, ?> twoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        assertTrue(twoBitQuantizer instanceof MultiBitScalarQuantizer);
        assertEquals(2, ((MultiBitScalarQuantizer) twoBitQuantizer).getBitsPerCoordinate());

        // Test for MultiBitScalarQuantizer (4-bit)
        ScalarQuantizationParams fourBitParams = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        Quantizer<?, ?> fourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        assertTrue(fourBitQuantizer instanceof MultiBitScalarQuantizer);
        assertEquals(4, ((MultiBitScalarQuantizer) fourBitQuantizer).getBitsPerCoordinate());
    }

    public void testQuantizerRegistryIsSingleton() {
        // Ensure the same instance is returned for the same type identifier
        ScalarQuantizationParams oneBitParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        Quantizer<?, ?> firstOneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        Quantizer<?, ?> secondOneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        assertSame(firstOneBitQuantizer, secondOneBitQuantizer);

        // Ensure the same instance is returned for the same type identifier (2-bit)
        ScalarQuantizationParams twoBitParams = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        Quantizer<?, ?> firstTwoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        Quantizer<?, ?> secondTwoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        assertSame(firstTwoBitQuantizer, secondTwoBitQuantizer);

        // Ensure the same instance is returned for the same type identifier (4-bit)
        ScalarQuantizationParams fourBitParams = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        Quantizer<?, ?> firstFourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        Quantizer<?, ?> secondFourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        assertSame(firstFourBitQuantizer, secondFourBitQuantizer);
    }

    public void testRegisterQuantizerThrowsExceptionWhenAlreadyRegistered() {
        ScalarQuantizationParams oneBitParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);

        // Attempt to register the same quantizer again should throw an exception
        assertThrows(IllegalArgumentException.class, () -> {
            QuantizerRegistry.register(
                    ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.ONE_BIT),
                    new OneBitScalarQuantizer()
            );
        });
    }
}
