/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The QuantizerRegistry class is responsible for managing the registration and retrieval
 * of quantizer instances. Quantizers are registered with specific quantization parameters
 * and type identifiers, allowing for efficient lookup and instantiation.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class QuantizerRegistry {
    // ConcurrentHashMap for thread-safe access
    private static final Map<String, Quantizer<?, ?>> registry = new ConcurrentHashMap<>();

    /**
     * Registers a quantizer with the registry.
     *
     * @param paramIdentifier the unique identifier for the quantization parameters
     * @param quantizer       an instance of the quantizer
     */
    static void register(final String paramIdentifier, final Quantizer<?, ?> quantizer) {
        // Ensure that the quantizer for this identifier is registered only once
        registry.putIfAbsent(paramIdentifier, quantizer);
    }

    /**
     * Retrieves a quantizer instance based on the provided quantization parameters.
     *
     * @param params the quantization parameters used to determine the appropriate quantizer
     * @param <P>    the type of quantization parameters
     * @param <Q>    the type of the quantized output
     * @return an instance of {@link Quantizer} corresponding to the provided parameters
     * @throws IllegalArgumentException if no quantizer is registered for the given parameters
     */
    static <P extends QuantizationParams, Q> Quantizer<P, Q> getQuantizer(final P params) {
        String identifier = params.getTypeIdentifier();
        Quantizer<?, ?> quantizer = registry.get(identifier);
        if (quantizer == null) {
            throw new IllegalArgumentException("No quantizer registered for type identifier: " + identifier);
        }
        @SuppressWarnings("unchecked")
        Quantizer<P, Q> typedQuantizer = (Quantizer<P, Q>) quantizer;
        return typedQuantizer;
    }
}
