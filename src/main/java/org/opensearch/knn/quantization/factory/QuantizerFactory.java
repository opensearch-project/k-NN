/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.factory;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The QuantizerFactory class is responsible for creating instances of {@link Quantizer}
 * based on the provided {@link QuantizationParams}. It uses a registry to look up the
 * appropriate quantizer implementation for the given quantization parameters.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class QuantizerFactory {
    private static final AtomicBoolean isRegistered = new AtomicBoolean(false);

    /**
     * Retrieves a quantizer instance based on the provided quantization parameters.
     *
     * @param params the quantization parameters used to determine the appropriate quantizer
     * @param <P>    the type of quantization parameters, extending {@link QuantizationParams}
     * @param <T>    the type of the input vector to be quantized
     * @param <R>    the type of the output after quantization
     * @return an instance of {@link Quantizer} corresponding to the provided parameters
     */
    public static <P extends QuantizationParams, T, R> Quantizer<T, R> getQuantizer(final P params) {
        if (params == null) {
            throw new IllegalArgumentException("Quantization parameters must not be null.");
        }
        // Lazy Registration instead of static block as class level;
        ensureRegistered();
        return (Quantizer<T, R>) QuantizerRegistry.getQuantizer(params);
    }

    /**
     * Ensures that default quantizers are registered.
     */
    private static void ensureRegistered() {
        if (!isRegistered.get()) {
            synchronized (QuantizerFactory.class) {
                if (!isRegistered.get()) {
                    QuantizerRegistrar.registerDefaultQuantizers();
                    isRegistered.set(true);
                }
            }
        }
    }
}
