/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.util.QuantizationStateSerializer;

import java.io.IOException;

/**
 * OneBitScalarQuantizationState represents the state of one-bit scalar quantization,
 * including the mean values used for quantization.
 */
public final class OneBitScalarQuantizationState implements QuantizationState {
    private final SQParams quantizationParams;
    private final float[] meanThresholds;

    /**
     * Constructs a OneBitScalarQuantizationState with the given quantization parameters and mean values.
     *
     * @param quantizationParams the scalar quantization parameters.
     * @param mean               the mean values for each dimension.
     */
    public OneBitScalarQuantizationState(final SQParams quantizationParams, final float[] mean) {
        this.quantizationParams = quantizationParams;
        this.meanThresholds = mean;
    }

    @Override
    public SQParams getQuantizationParams() {
        return quantizationParams;
    }

    /**
     * Returns the mean values used in the quantization process.
     *
     * @return an array of mean values.
     */
    public float[] getMeanThresholds() {
        return meanThresholds;
    }

    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this, meanThresholds);
    }

    public static OneBitScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException, ClassNotFoundException {
        return (OneBitScalarQuantizationState) QuantizationStateSerializer.deserialize(
            bytes,
            (parentParams, meanThresholds) -> new OneBitScalarQuantizationState((SQParams) parentParams, (float[]) meanThresholds)
        );
    }
}
