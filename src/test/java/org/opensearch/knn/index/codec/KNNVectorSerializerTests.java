/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.codec.util.SerializationMode;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.stream.IntStream;

public class KNNVectorSerializerTests extends KNNTestCase {

    Random random = new Random();

    public void testVectorSerializerFactory() throws Exception {
        //check that default serializer can work with array of floats
        //setup
        final float[] vector = getArrayOfRandomFloats(20);
        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector)
            ds.writeFloat(f);
        final byte[] vectorAsCollectionOfFloats = bas.toByteArray();
        final ByteArrayInputStream bais = new ByteArrayInputStream(vectorAsCollectionOfFloats);
        bais.reset();

        final KNNVectorSerializer defaultSerializer = KNNVectorSerializerFactory.getDefaultSerializer();
        assertNotNull(defaultSerializer);

        final float[] actualDeserializedVector = defaultSerializer.byteToFloatArray(bais);
        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);

        final KNNVectorSerializer arraySerializer =
                KNNVectorSerializerFactory.getSerializerBySerializationMode(SerializationMode.ARRAY);
        assertNotNull(arraySerializer);

        final KNNVectorSerializer collectionOfFloatsSerializer =
                KNNVectorSerializerFactory.getSerializerBySerializationMode(SerializationMode.COLLECTION_OF_FLOATS);
        assertNotNull(collectionOfFloatsSerializer);
    }


    public void testVectorSerializerFactory_throwExceptionForStreamWithUnsupportedDataType() throws Exception {
        //prepare array of chars that is not supported by serializer factory. expected behavior is to fail
        final char[] arrayOfChars = new char[] {'a', 'b', 'c'};
        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (char ch : arrayOfChars)
            ds.writeChar(ch);
        final byte[] vectorAsCollectionOfChars = bas.toByteArray();
        final ByteArrayInputStream bais = new ByteArrayInputStream(vectorAsCollectionOfChars);
        bais.reset();

        expectThrows(RuntimeException.class, () -> KNNVectorSerializerFactory.getSerializerByStreamContent(bais));
    }

    public void testVectorAsArraySerializer() throws Exception {
        final float[] vector = getArrayOfRandomFloats(20);

        final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        final ObjectOutputStream objectStream = new ObjectOutputStream(byteStream);
        objectStream.writeObject(vector);
        final byte[] serializedVector = byteStream.toByteArray();
        final ByteArrayInputStream bais = new ByteArrayInputStream(serializedVector);

        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(bais);

        //testing serialization
        bais.reset();
        final byte[] actualSerializedVector = vectorSerializer.floatToByteArray(vector);

        assertNotNull(actualSerializedVector);
        assertArrayEquals(serializedVector, actualSerializedVector);

        //testing deserialization
        bais.reset();
        final float[] actualDeserializedVector = vectorSerializer.byteToFloatArray(bais);

        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);
    }

    public void testVectorAsCollectionOfFloatsSerializer() throws Exception {
        //setup
        final float[] vector = getArrayOfRandomFloats(20);

        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector)
            ds.writeFloat(f);
        final byte[] vectorAsCollectionOfFloats = bas.toByteArray();
        final ByteArrayInputStream bais = new ByteArrayInputStream(vectorAsCollectionOfFloats);

        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(bais);

        //testing serialization
        bais.reset();
        final byte[] actualSerializedVector = vectorSerializer.floatToByteArray(vector);

        assertNotNull(actualSerializedVector);
        assertArrayEquals(vectorAsCollectionOfFloats, actualSerializedVector);

        //testing deserialization
        bais.reset();
        final float[] actualDeserializedVector = vectorSerializer.byteToFloatArray(bais);

        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);
    }

    private float[] getArrayOfRandomFloats(int arrayLength) {
        float[] vector = new float[arrayLength];
        IntStream.range(0, arrayLength).forEach(index -> vector[index] = random.nextFloat());
        return vector;
    }
}
