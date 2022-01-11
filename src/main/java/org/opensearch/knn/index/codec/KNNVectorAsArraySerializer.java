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

package org.opensearch.knn.index.codec;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Class implements KNNVectorSerializer based on standard Java serialization/deserialization as a single array object
 */
public class KNNVectorAsArraySerializer implements KNNVectorSerializer {
    @Override
    public byte[] floatToByteArray(float[] input) throws Exception {
        byte[] bytes;
        try (ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
             ObjectOutputStream objectStream = new ObjectOutputStream(byteStream);) {
            objectStream.writeObject(input);
            bytes = byteStream.toByteArray();
        }
        return bytes;
    }

    @Override
    public float[] byteToFloatArray(ByteArrayInputStream byteStream) throws IOException, ClassNotFoundException {
        final ObjectInputStream objectStream = new ObjectInputStream(byteStream);
        final float[] vector = (float[]) objectStream.readObject();
        return vector;
    }
}
