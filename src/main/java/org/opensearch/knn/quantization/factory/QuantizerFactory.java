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

import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.Quantizer;

public class QuantizerFactory {
    static {
        // Register all quantizers here
        QuantizerRegistry.register(SQParams.class, SQTypes.ONE_BIT.name(), OneBitScalarQuantizer::new);
    }

    public static Quantizer<?, ?> getQuantizer(QuantizationParams params) {
        if (params instanceof SQParams) {
            SQParams sqParams = (SQParams) params;
            return QuantizerRegistry.getQuantizer(params, sqParams.getSqType().name());
        }
        // Add more cases for other quantization parameters here
        throw new IllegalArgumentException("Unsupported quantization parameters: " + params.getClass().getName());
    }
}
