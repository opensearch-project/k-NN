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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.stream.IntStream;

/**
 * Class implements VectorSerializer based on standard Java serialization/deserialization as a single array object
 */
public class VectorAsCollectionOfFloatsSerializer implements VectorSerializer{
    private static final int BYTES_IN_FLOAT = 4;

    @Override
    public byte[] floatToByte(float[] floats) {
        final ByteBuffer bb = ByteBuffer.allocate(floats.length * BYTES_IN_FLOAT).order(ByteOrder.BIG_ENDIAN);
        IntStream.range(0, floats.length).forEach((index) -> bb.putFloat(floats[index]));
        byte[] bytes = new byte[bb.flip().limit()];
        bb.get(bytes);
        return bytes;
    }

    @Override
    public float[] byteToFloat(ByteArrayInputStream byteStream) {
        final byte[] vectorAsByteArray = new byte[byteStream.available()];
        byteStream.read(vectorAsByteArray, 0, byteStream.available());
        final float[] vector = new float[vectorAsByteArray.length / BYTES_IN_FLOAT];
        ByteBuffer.wrap(vectorAsByteArray).asFloatBuffer().get(vector);
        return vector;
    }
}