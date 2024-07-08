/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.factory;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.Quantizer;
import org.junit.BeforeClass;

public class QuantizerRegistryTests extends KNNTestCase {
    @BeforeClass
    public static void setup() {
        // Register the quantizer for testing
        QuantizerRegistry.register(SQParams.class, SQTypes.ONE_BIT.name(), OneBitScalarQuantizer::new);
    }

    public void testRegisterAndGetQuantizer() {
        SQParams params = new SQParams(SQTypes.ONE_BIT);
        Quantizer<?, ?> quantizer = QuantizerRegistry.getQuantizer(params, SQTypes.ONE_BIT.name());
        assertTrue(quantizer instanceof OneBitScalarQuantizer);
    }

    public void testGetQuantizer_withUnsupportedTypeIdentifier() {
        SQParams params = new SQParams(SQTypes.ONE_BIT);
        expectThrows( IllegalArgumentException.class, ()-> QuantizerRegistry.getQuantizer(params, "UNSUPPORTED_TYPE"));
    }
}
