/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import org.opensearch.knn.quantization.enums.QuantizationType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * The QuantizerRegistry class is responsible for managing the registration and retrieval
 * of quantizer instances. Quantizers are registered with specific quantization parameters
 * and type identifiers, allowing for efficient lookup and instantiation.
 */
final class QuantizerRegistry {

    // Private constructor to prevent instantiation
    private QuantizerRegistry() {}

    // ConcurrentHashMap for thread-safe access
    private static final Map<String, Supplier<? extends Quantizer<?, ?>>> registry = new ConcurrentHashMap<>();

    /**
     * Registers a quantizer with the registry.
     *
     * @param paramClass        the class of the quantization parameters
     * @param quantizationType  the quantization type (e.g., VALUE_QUANTIZATION)
     * @param sqType            the specific quantization subtype (e.g., ONE_BIT, TWO_BIT)
     * @param quantizerSupplier a supplier that provides instances of the quantizer
     * @param <P>               the type of quantization parameters
     */
    public static <P extends QuantizationParams> void register(
        final Class<P> paramClass,
        final QuantizationType quantizationType,
        final ScalarQuantizationType sqType,
        final Supplier<? extends Quantizer<?, ?>> quantizerSupplier
    ) {
        String identifier = createIdentifier(quantizationType, sqType);
        // Ensure that the quantizer for this identifier is registered only once
        registry.computeIfAbsent(identifier, key -> quantizerSupplier);
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
    public static <P extends QuantizationParams, Q> Quantizer<P, Q> getQuantizer(final P params) {
        String identifier = params.getTypeIdentifier();
        Supplier<? extends Quantizer<?, ?>> supplier = registry.get(identifier);
        if (supplier == null) {
            throw new IllegalArgumentException(
                "No quantizer registered for type identifier: " + identifier + ". Available quantizers: " + registry.keySet()
            );
        }
        @SuppressWarnings("unchecked")
        Quantizer<P, Q> quantizer = (Quantizer<P, Q>) supplier.get();
        return quantizer;
    }

    /**
     * Creates a unique identifier for the quantizer based on the quantization type and specific quantization subtype.
     *
     * @param quantizationType the quantization type
     * @param sqType           the specific quantization subtype
     * @return a string identifier
     */
    private static String createIdentifier(final QuantizationType quantizationType, final ScalarQuantizationType sqType) {
        return quantizationType.name() + "_" + sqType.name();
    }
}
