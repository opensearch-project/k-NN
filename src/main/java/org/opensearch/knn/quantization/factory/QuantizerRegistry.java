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

import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

public class QuantizerRegistry {
    private static final Map<Class<? extends QuantizationParams>, Map<String, Supplier<? extends Quantizer<?, ?>>>> registry = new HashMap<>();

    public static <T extends QuantizationParams> void register(Class<T> paramClass, String typeIdentifier, Supplier<? extends Quantizer<?, ?>> quantizerSupplier) {
        registry.computeIfAbsent(paramClass, k -> new HashMap<>()).put(typeIdentifier, quantizerSupplier);
    }

    public static Quantizer<?, ?> getQuantizer(QuantizationParams params, String typeIdentifier) {
        Map<String, Supplier<? extends Quantizer<?, ?>>> typeMap = registry.get(params.getClass());
        if (typeMap == null) {
            throw new IllegalArgumentException("No quantizer registered for parameters: " + params.getClass().getName());
        }
        Supplier<? extends Quantizer<?, ?>> supplier = typeMap.get(typeIdentifier);
        if (supplier == null) {
            throw new IllegalArgumentException("No quantizer registered for type identifier: " + typeIdentifier);
        }
        return supplier.get();
    }
}
