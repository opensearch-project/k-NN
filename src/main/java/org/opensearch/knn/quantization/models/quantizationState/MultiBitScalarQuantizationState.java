/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.util.QuantizationStateSerializer;

import java.io.IOException;

/**
 * MultiBitScalarQuantizationState represents the state of multi-bit scalar quantization,
 * including the thresholds used for quantization.
 */
public final class MultiBitScalarQuantizationState implements QuantizationState {
    private final SQParams quantizationParams;
    private final float[][] thresholds;

    /**
     * Constructs a MultiBitScalarQuantizationState with the given quantization parameters and thresholds.
     *
     * @param quantizationParams the scalar quantization parameters.
     * @param thresholds         the threshold values for multi-bit quantization, organized as a 2D array
     *                           where each row corresponds to a different bit level.
     */
    public MultiBitScalarQuantizationState(final SQParams quantizationParams, final float[][] thresholds) {
        this.quantizationParams = quantizationParams;
        this.thresholds = thresholds;
    }

    @Override
    public SQParams getQuantizationParams() {
        return quantizationParams;
    }

    /**
     * Returns the thresholds used in the quantization process.
     *
     * @return a 2D array of threshold values.
     */
    public float[][] getThresholds() {
        return thresholds;
    }

    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this, thresholds);
    }

    public static MultiBitScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException, ClassNotFoundException {
        return (MultiBitScalarQuantizationState) QuantizationStateSerializer.deserialize(
            bytes,
            (parentParams, thresholds) -> new MultiBitScalarQuantizationState((SQParams) parentParams, (float[][]) thresholds)
        );
    }
}
