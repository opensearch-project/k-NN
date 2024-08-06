/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import org.junit.BeforeClass;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.QuantizationType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.quantizer.MultiBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.Quantizer;

public class QuantizerRegistryTests extends KNNTestCase {

    @BeforeClass
    public static void setup() {
        // Register the quantizers for testing with enums
        QuantizerRegistry.register(SQParams.class, QuantizationType.VALUE, ScalarQuantizationType.ONE_BIT, OneBitScalarQuantizer::new);
        QuantizerRegistry.register(
            SQParams.class,
            QuantizationType.VALUE,
            ScalarQuantizationType.TWO_BIT,
            () -> new MultiBitScalarQuantizer(2)
        );
        QuantizerRegistry.register(
            SQParams.class,
            QuantizationType.VALUE,
            ScalarQuantizationType.FOUR_BIT,
            () -> new MultiBitScalarQuantizer(4)
        );
    }

    public void testRegisterAndGetQuantizer() {
        // Test for OneBitScalarQuantizer
        SQParams oneBitParams = new SQParams(ScalarQuantizationType.ONE_BIT);
        Quantizer<?, ?> oneBitQuantizer = QuantizerRegistry.getQuantizer(oneBitParams);
        assertTrue(oneBitQuantizer instanceof OneBitScalarQuantizer);

        // Test for MultiBitScalarQuantizer (2-bit)
        SQParams twoBitParams = new SQParams(ScalarQuantizationType.TWO_BIT);
        Quantizer<?, ?> twoBitQuantizer = QuantizerRegistry.getQuantizer(twoBitParams);
        assertTrue(twoBitQuantizer instanceof MultiBitScalarQuantizer);

        // Test for MultiBitScalarQuantizer (4-bit)
        SQParams fourBitParams = new SQParams(ScalarQuantizationType.FOUR_BIT);
        Quantizer<?, ?> fourBitQuantizer = QuantizerRegistry.getQuantizer(fourBitParams);
        assertTrue(fourBitQuantizer instanceof MultiBitScalarQuantizer);
    }

    public void testGetQuantizer_withUnsupportedTypeIdentifier() {
        // Create SQParams with an unsupported type identifier
        SQParams params = new SQParams(ScalarQuantizationType.UNSUPPORTED_TYPE); // Assuming UNSUPPORTED_TYPE is not registered

        // Expect IllegalArgumentException when requesting a quantizer with unsupported params
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> { QuantizerRegistry.getQuantizer(params); }
        );

        assertTrue(exception.getMessage().contains("No quantizer registered for type identifier"));
    }
}
