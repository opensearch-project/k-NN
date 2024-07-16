/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.KNNTestCase;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.stream.IntStream;

public class KNNVectorSerializerTests extends KNNTestCase {

    Random random = new Random();

    public void testVectorSerializerFactory() throws Exception {
        // check that default serializer can work with array of floats
        // setup
        final float[] vector = getArrayOfRandomFloats(20);
        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector)
            ds.writeFloat(f);
        final BytesRef vectorAsCollectionOfFloats = new BytesRef(bas.toByteArray());
        final KNNVectorSerializer defaultSerializer = KNNVectorSerializerFactory.getDefaultSerializer();
        assertNotNull(defaultSerializer);

        final float[] actualDeserializedVector = defaultSerializer.byteToFloatArray(vectorAsCollectionOfFloats);
        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);

        final KNNVectorSerializer arraySerializer = KNNVectorSerializerFactory.getSerializerBySerializationMode(SerializationMode.ARRAY);
        assertNotNull(arraySerializer);

        final KNNVectorSerializer collectionOfFloatsSerializer = KNNVectorSerializerFactory.getSerializerBySerializationMode(
            SerializationMode.COLLECTION_OF_FLOATS
        );
        assertNotNull(collectionOfFloatsSerializer);
    }

    public void testVectorSerializerFactory_throwExceptionForBytesWithUnsupportedDataType() throws Exception {
        // prepare array of chars that is not supported by serializer factory. expected behavior is to fail
        final char[] arrayOfChars = new char[] { 'a', 'b', 'c' };
        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (char ch : arrayOfChars)
            ds.writeChar(ch);
        final BytesRef vectorAsCollectionOfChars = new BytesRef(bas.toByteArray());

        expectThrows(RuntimeException.class, () -> KNNVectorSerializerFactory.getSerializerByBytesRef(vectorAsCollectionOfChars));
    }

    public void testVectorAsArraySerializer() throws Exception {
        final float[] vector = getArrayOfRandomFloats(20);

        final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        final ObjectOutputStream objectStream = new ObjectOutputStream(byteStream);
        objectStream.writeObject(vector);
        final BytesRef serializedVector = new BytesRef(byteStream.toByteArray());
        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByBytesRef(serializedVector);

        // testing serialization
        final byte[] actualSerializedVector = vectorSerializer.floatToByteArray(vector);

        assertNotNull(actualSerializedVector);
        assertArrayEquals(serializedVector.bytes, actualSerializedVector);

        // testing deserialization
        final float[] actualDeserializedVector = vectorSerializer.byteToFloatArray(serializedVector);

        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);
    }

    public void testVectorAsCollectionOfFloatsSerializer() throws Exception {
        // setup
        final float[] vector = getArrayOfRandomFloats(20);

        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector)
            ds.writeFloat(f);
        final BytesRef vectorAsCollectionOfFloats = new BytesRef(bas.toByteArray());
        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByBytesRef(vectorAsCollectionOfFloats);

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
        for (SerializationMode serializationMode : SerializationMode.values()) {
            final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerBySerializationMode(serializationMode);
            assertNotNull(vectorSerializer);
            byte[] bytes = vectorSerializer.floatToByteArray(vector);
            byte[] bytesWithOffset = new byte[bytes.length + 2 * offset];
            System.arraycopy(bytes, 0, bytesWithOffset, offset, bytes.length);
            BytesRef serializedVector = new BytesRef(bytesWithOffset, offset, bytes.length);
            float[] deserializedVector = vectorSerializer.byteToFloatArray(serializedVector);
            assertArrayEquals(vector, deserializedVector, 0.1f);
        }
    }

    private float[] getArrayOfRandomFloats(int arrayLength) {
        float[] vector = new float[arrayLength];
        IntStream.range(0, arrayLength).forEach(index -> vector[index] = random.nextFloat());
        return vector;
    }
}
