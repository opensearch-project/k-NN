/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.quantizer.MultiBitScalarQuantizer;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;

/**
 * The QuantizerRegistrar class is responsible for registering default quantizers.
 * This class ensures that the registration happens only once in a thread-safe manner.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class QuantizerRegistrar {

    /**
     * Registers default quantizers
     * <p>
     * This method is synchronized to ensure that registration occurs only once,
     * even in a multi-threaded environment.
     * </p>
     */
    static synchronized void registerDefaultQuantizers() {
        // Register OneBitScalarQuantizer for SQParams with VALUE_QUANTIZATION and SQTypes.ONE_BIT
        QuantizerRegistry.register(
            ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.ONE_BIT),
            new OneBitScalarQuantizer()
        );
        // Register MultiBitScalarQuantizer for SQParams with VALUE_QUANTIZATION with bit per co-ordinate = 2
        QuantizerRegistry.register(
            ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.TWO_BIT),
            new MultiBitScalarQuantizer(2)
        );
        // Register MultiBitScalarQuantizer for SQParams with VALUE_QUANTIZATION with bit per co-ordinate = 4
        QuantizerRegistry.register(
            ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.FOUR_BIT),
            new MultiBitScalarQuantizer(4)
        );
        // Register ByteScalarQuantizer for SQParams with int8 or 8 bits
        // QuantizerRegistry.register(
        // ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.EIGHT_BIT),
        // new ByteScalarQuantizer(8)
        // );
    }
}
