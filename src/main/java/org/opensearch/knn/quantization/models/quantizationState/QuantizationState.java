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

import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;

public abstract class QuantizationState implements Serializable {
    private QuantizationParams quantizationParams;

    public QuantizationState(QuantizationParams quantizationParams) {
        this.quantizationParams = quantizationParams;
    }

    public QuantizationParams getQuantizationParams() {
        return quantizationParams;
    }

    public byte[] toByteArray() throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(bos);
        out.writeObject(this);
        out.flush();
        return bos.toByteArray();
    }

    public static QuantizationState fromByteArray(byte[] bytes) throws IOException, ClassNotFoundException {
        ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
        ObjectInputStream in = new ObjectInputStream(bis);
        return (QuantizationState) in.readObject();
    }
}
