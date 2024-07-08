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

public class QuantizerFactoryTests extends KNNTestCase {
    public void testGetQuantizer_withSQParams() {
        SQParams params = new SQParams(SQTypes.ONE_BIT);
        Quantizer<?, ?> quantizer = QuantizerFactory.getQuantizer(params);
        assertTrue(quantizer instanceof OneBitScalarQuantizer);
    }
}
