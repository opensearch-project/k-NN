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

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.quantizationOutput.OneBitScalarQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.SamplingTrainingRequest;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

public class OneBitScalarQuantizer implements Quantizer<float[], byte[]> {
    private static final int SAMPLING_SIZE = 25000;

    @Override
    public int getSamplingSize() {
        return SAMPLING_SIZE;
    }

    @Override
    public QuantizationState train(TrainingRequest<float[]> trainingRequest) {
        if (!(trainingRequest instanceof SamplingTrainingRequest)) {
            throw new IllegalArgumentException("Training request must be of type SamplingTrainingRequest.");
        }

        SamplingTrainingRequest<float[]> samplingRequest = (SamplingTrainingRequest<float[]>) trainingRequest;
        int[] sampledIndices = samplingRequest.getSampledIndices();

        if (sampledIndices == null || sampledIndices.length == 0) {
            throw new IllegalArgumentException("Sampled indices must not be null or empty.");
        }

        int totalSamples = sampledIndices.length;
        float[] sum = null;

        // Calculate the sum for each dimension based on sampled indices
        for (int i = 0; i < totalSamples; i++) {
            float[] vector = samplingRequest.getVectorByDocId(sampledIndices[i]);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + sampledIndices[i] + " is null.");
            }
            if (sum == null) {
                sum = new float[vector.length];
            } else if (sum.length != vector.length) {
                throw new IllegalArgumentException("All vectors must have the same dimension.");
            }
            for (int j = 0; j < vector.length; j++) {
                sum[j] += vector[j];
            }
        }
        if (sum == null) {
            throw new IllegalStateException("Sum array should not be null after processing vectors.");
        }
        // Calculate the mean for each dimension
        float[] mean = new float[sum.length];
        for (int j = 0; j < sum.length; j++) {
            mean[j] = sum[j] / totalSamples;
        }
        SQParams params = (SQParams) trainingRequest.getParams();
        if (params == null) {
            throw new IllegalArgumentException("Quantization parameters must not be null.");
        }
        return new OneBitScalarQuantizationState(params, mean);
    }

    @Override
    public QuantizationOutput<byte[]> quantize(float[] vector, QuantizationState state) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        if (!(state instanceof OneBitScalarQuantizationState)) {
            throw new IllegalArgumentException("Quantization state must be of type OneBitScalarQuantizationState.");
        }
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;
        float[] thresholds = binaryState.getMean();
        if (thresholds == null || thresholds.length != vector.length) {
            throw new IllegalArgumentException("Thresholds must not be null and must match the dimension of the vector.");
        }
        byte[] quantizedVector = new byte[vector.length];
        for (int i = 0; i < vector.length; i++) {
            quantizedVector[i] = (byte) (vector[i] > thresholds[i] ? 1 : 0);
        }
        return new OneBitScalarQuantizationOutput(packBitsFromBitArray(quantizedVector));
    }

    @Override
    public QuantizationOutput<byte[]> quantize(float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        byte[] quantizedVector = new byte[vector.length];
        for (int i = 0; i < vector.length; i++) {
            quantizedVector[i] = (byte) (vector[i] > 0 ? 1 : 0);
        }
        return new OneBitScalarQuantizationOutput(packBitsFromBitArray(quantizedVector));
    }

    private  byte[] packBitsFromBitArray(byte[] bitArray) {
        int bitLength = bitArray.length;
        int byteLength = (bitLength + 7) / 8;
        byte[] packedArray = new byte[byteLength];

        for (int i = 0; i < bitLength; i++) {
            if (bitArray[i] != 0 && bitArray[i] != 1) {
                throw new IllegalArgumentException("Array elements must be 0 or 1");
            }
            int byteIndex = i / 8;
            int bitIndex = 7 - (i % 8);
            packedArray[byteIndex] |= (bitArray[i] << bitIndex);
        }

        return packedArray;
    }
}

