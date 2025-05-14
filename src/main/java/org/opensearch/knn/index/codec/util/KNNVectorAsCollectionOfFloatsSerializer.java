/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Class implements KNNVectorSerializer based on serialization/deserialization of array as collection of individual numbers
 */
public class KNNVectorAsCollectionOfFloatsSerializer implements KNNVectorSerializer {
    private static final int BYTES_IN_FLOAT = 4;

    public static final KNNVectorAsCollectionOfFloatsSerializer INSTANCE = new KNNVectorAsCollectionOfFloatsSerializer();

    @Override
    public byte[] floatToByteArray(float[] input) {
        final ByteBuffer bb = ByteBuffer.allocate(input.length * BYTES_IN_FLOAT).order(ByteOrder.BIG_ENDIAN);
        IntStream.range(0, input.length).forEach((index) -> bb.putFloat(input[index]));
        byte[] bytes = new byte[bb.flip().limit()];
        bb.get(bytes);
        return bytes;
    }

    @Override
    public float[] byteToFloatArray(BytesRef bytesRef) {
        if (bytesRef == null || bytesRef.length % BYTES_IN_FLOAT != 0) {
            throw new IllegalArgumentException("Byte stream cannot be deserialized to array of floats");
        }
        final int sizeOfFloatArray = bytesRef.length / BYTES_IN_FLOAT;
        final float[] vector = new float[sizeOfFloatArray];
        ByteBuffer.wrap(bytesRef.bytes, bytesRef.offset, bytesRef.length).asFloatBuffer().get(vector);
        return vector;
    }

    @Override
    public ByteBuffer floatsToByteArray(List<float[]> input) {
        //TODO
        if (input.isEmpty()) {
            return null;
        }
        int bufferSize = input.size() * input.get(0).length * BYTES_IN_FLOAT;
        final ByteBuffer bb = ByteBuffer.allocate(bufferSize).order(ByteOrder.BIG_ENDIAN);

        for (float[] vector : input) {
            for(float f : vector) {
                bb.putFloat(f);
            }
        }
        return bb;
    }

    @Override
    public List<float[]> byteToFloatsArray(BytesRef bytesRef, int dims) {
        if (bytesRef == null || bytesRef.length % BYTES_IN_FLOAT != 0) {
            throw new IllegalArgumentException("Byte stream cannot be deserialized to arrays of floats");
        }
        final int sizeOfFloatArray = bytesRef.length / BYTES_IN_FLOAT;
        int number_vectors = sizeOfFloatArray / dims;
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0, offset = 0; i < number_vectors; i++) {
            final float[] vector = new float[dims];
            ByteBuffer.wrap(bytesRef.bytes, offset, dims * BYTES_IN_FLOAT).asFloatBuffer().get(vector);
            vectors.add(vector);
            offset += dims * BYTES_IN_FLOAT;
        }
        return vectors;
    }
}
