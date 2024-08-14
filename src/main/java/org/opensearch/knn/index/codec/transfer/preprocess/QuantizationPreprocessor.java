/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer.preprocess;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.quantizer.Quantizer;

@RequiredArgsConstructor
public class QuantizationPreprocessor<T, V> implements PreprocessVectorTransfer<T, V> {

    private final Quantizer<T, V> quantizer;
    private final QuantizationState quantizationState;

    @Getter
    private final QuantizationOutput<V> quantizationOutput;

    @Override
    public V apply(T vector) {
        quantizer.quantize(vector, quantizationState, quantizationOutput);
        return quantizationOutput.getQuantizedVector();
    }
}
