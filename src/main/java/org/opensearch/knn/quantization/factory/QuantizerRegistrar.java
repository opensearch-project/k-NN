/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import org.opensearch.knn.quantization.enums.QuantizationType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.quantizer.MultiBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;

/**
 * The QuantizerRegistrar class is responsible for registering default quantizers.
 * This class ensures that the registration happens only once in a thread-safe manner.
 */
final class QuantizerRegistrar {

    // Private constructor to prevent instantiation
    private QuantizerRegistrar() {}

    /**
     * Registers default quantizers if not already registered.
     * <p>
     * This method is synchronized to ensure that registration occurs only once,
     * even in a multi-threaded environment.
     * </p>
     */
    public static synchronized void registerDefaultQuantizers() {
        // Register OneBitScalarQuantizer for SQParams with VALUE_QUANTIZATION and SQTypes.ONE_BIT
        QuantizerRegistry.register(SQParams.class, QuantizationType.VALUE, ScalarQuantizationType.ONE_BIT, OneBitScalarQuantizer::new);
        // Register MultiBitScalarQuantizer for SQParams with VALUE_QUANTIZATION with bit per co-ordinate = 2
        QuantizerRegistry.register(
            SQParams.class,
            QuantizationType.VALUE,
            ScalarQuantizationType.TWO_BIT,
            () -> new MultiBitScalarQuantizer(2)
        );
        // Register MultiBitScalarQuantizer for SQParams with VALUE_QUANTIZATION with bit per co-ordinate = 4
        QuantizerRegistry.register(
            SQParams.class,
            QuantizationType.VALUE,
            ScalarQuantizationType.FOUR_BIT,
            () -> new MultiBitScalarQuantizer(4)
        );
    }
}
