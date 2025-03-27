/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.KNNTestCase;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.util.Random;
import java.util.stream.IntStream;

public class KNNVectorSerializerTests extends KNNTestCase {

    Random random = new Random();

    public void testVectorAsCollectionOfFloatsSerializer() throws Exception {
        // setup
        final float[] vector = getArrayOfRandomFloats(20);

        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector)
            ds.writeFloat(f);
        final BytesRef vectorAsCollectionOfFloats = new BytesRef(bas.toByteArray());
        final KNNVectorSerializer vectorSerializer = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE;

        // testing serialization
        final byte[] actualSerializedVector = vectorSerializer.floatToByteArray(vector);

        assertNotNull(actualSerializedVector);
        assertArrayEquals(vectorAsCollectionOfFloats.bytes, actualSerializedVector);

        // testing deserialization
        final float[] actualDeserializedVector = vectorSerializer.byteToFloatArray(vectorAsCollectionOfFloats);

        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);
    }

    public void testVectorSerializer_whenVectorBytesOffset_thenSuccess() {
        final float[] vector = getArrayOfRandomFloats(20);
        int offset = randomInt(4);
        final KNNVectorSerializer vectorSerializer = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE;
        assertNotNull(vectorSerializer);
        byte[] bytes = vectorSerializer.floatToByteArray(vector);
        byte[] bytesWithOffset = new byte[bytes.length + 2 * offset];
        System.arraycopy(bytes, 0, bytesWithOffset, offset, bytes.length);
        BytesRef serializedVector = new BytesRef(bytesWithOffset, offset, bytes.length);
        float[] deserializedVector = vectorSerializer.byteToFloatArray(serializedVector);
        assertArrayEquals(vector, deserializedVector, 0.1f);
    }

    private float[] getArrayOfRandomFloats(int arrayLength) {
        float[] vector = new float[arrayLength];
        IntStream.range(0, arrayLength).forEach(index -> vector[index] = random.nextFloat());
        return vector;
    }
}
