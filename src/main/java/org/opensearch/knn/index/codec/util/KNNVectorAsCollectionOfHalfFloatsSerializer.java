/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;

import org.opensearch.knn.jni.JNICommons;

/**
 * Class implements KNNVectorSerializer based on serialization/deserialization of float array
 * as a collection of individual half-precision values
 */
public class KNNVectorAsCollectionOfHalfFloatsSerializer implements KNNVectorSerializer {
    private static final int BYTES_IN_HALF_FLOAT = 2;
    private final int dimension;
    private final byte[] bytesBuffer;
    private final float[] floatBuffer;

    /**
     * Constructs a serializer for vectors of the specified dimension.
     *
     * @param dimension the expected length of all float[] vectors to serialize/deserialize
     */
    public KNNVectorAsCollectionOfHalfFloatsSerializer(int dimension) {
        this.dimension = dimension;
        this.bytesBuffer = new byte[dimension * BYTES_IN_HALF_FLOAT];
        this.floatBuffer = new float[dimension];
    }

    /**
     * Converts float[] to byte[] using JNI (FP32 to FP16).
     * @param input the float[] to be serialized into half-precision format
     * @return a byte[] containing the float16-encoded data
     */
    @Override
    public byte[] floatToByteArray(float[] input) {
        if (input == null) {
            throw new IllegalArgumentException("Input float array is null. Cannot convert to FP16.");
        }
        if (input.length != dimension) {
            throw new IllegalArgumentException(
                "Input float array length (" + input.length + ") does not match expected dimension (" + dimension + ")."
            );
        }
        JNICommons.convertFP32ToFP16(input, bytesBuffer, input.length);
        return bytesBuffer;
    }

    /**
     * Converts a BytesRef-wrapped byte array (encoded as float16) back into a float array.
     *
     * @param bytesRef the BytesRef containing float16-encoded vector data
     * @return a float containing the decoded float32 values
     */
    @Override
    public float[] byteToFloatArray(BytesRef bytesRef) {
        if (bytesRef == null || bytesRef.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException("Byte stream cannot be deserialized to array of half-floats");
        }

        int count = bytesRef.length / BYTES_IN_HALF_FLOAT;
        if (count != dimension) {
            throw new IllegalArgumentException("Expected dimension " + dimension + " but got " + count);
        }

        JNICommons.convertFP16ToFP32(bytesRef.bytes, floatBuffer, count, bytesRef.offset);
        return floatBuffer;
    }

    /**
     * Converts byte[] to float[] using JNI (FP16 to FP32).
     * @param input the byte[] containing half-precision encoded data
     * @return a float[] containing the decoded float32 values
     */
    public float[] byteToFloatArray(byte[] input) {
        if (input == null) {
            throw new IllegalArgumentException("Input byte array is null. Cannot convert to FP32.");
        }
        if (input.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException(
                "Invalid byte array length ("
                    + input.length
                    + "). Must be a multiple of "
                    + BYTES_IN_HALF_FLOAT
                    + " to represent float16 values."
            );
        }

        int count = input.length / BYTES_IN_HALF_FLOAT;

        if (count != dimension) {
            throw new IllegalArgumentException(
                "Input byte array contains " + count + " float16 values, but expected dimension is " + dimension + "."
            );
        }

        JNICommons.convertFP16ToFP32(input, floatBuffer, count, 0);
        return floatBuffer;
    }
}
