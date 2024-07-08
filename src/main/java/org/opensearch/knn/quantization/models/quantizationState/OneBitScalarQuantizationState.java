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

package org.opensearch.knn.quantization.models.quantizationState;


import org.opensearch.knn.quantization.models.quantizationParams.SQParams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;

public class OneBitScalarQuantizationState extends QuantizationState {
    private float[] mean;

    public OneBitScalarQuantizationState(SQParams quantizationParams, float[] floatArray) {
        super(quantizationParams);
        this.mean = floatArray;
    }

    public float[] getMean() {
        return mean;
    }

    @Override
    public byte[] toByteArray() throws IOException {
        byte[] parentBytes = super.toByteArray();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(bos);
        out.write(parentBytes);
        out.writeObject(mean);
        out.flush();
        return bos.toByteArray();
    }

    public static OneBitScalarQuantizationState fromByteArray(byte[] bytes) throws IOException, ClassNotFoundException {
        ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
        ObjectInputStream in = new ObjectInputStream(bis);
        QuantizationState parentState = (QuantizationState) in.readObject();
        float[] floatArray = (float[]) in.readObject();
        return new OneBitScalarQuantizationState((SQParams) parentState.getQuantizationParams(), floatArray);
    }
}

