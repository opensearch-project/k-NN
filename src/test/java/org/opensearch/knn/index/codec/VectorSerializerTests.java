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

import org.opensearch.knn.KNNTestCase;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.stream.IntStream;

public class VectorSerializerTests extends KNNTestCase {

    Random random = new Random();

    public void testVectorSerializerFactory() {
        final VectorSerializer arraySerializer =
                VectorSerializerFactory.getSerializerBySerializationMode(VectorSerializerFactory.SerializationMode.ARRAY);
        assertNotNull(arraySerializer);

        final VectorSerializer collectionOfFloatsSerializer =
                VectorSerializerFactory.getSerializerBySerializationMode(VectorSerializerFactory.SerializationMode.COLLECTION_OF_FLOATS);
        assertNotNull(collectionOfFloatsSerializer);
    }

    public void testVectorAsArraySerializer() throws Exception {
        int arrayLength = 20;
        float[] vector = new float[arrayLength];
        IntStream.range(0, arrayLength).forEach(index -> vector[index] = random.nextFloat());

        final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        final ObjectOutputStream objectStream = new ObjectOutputStream(byteStream);
        objectStream.writeObject(vector);
        final byte[] serializedVector = byteStream.toByteArray();
        final ByteArrayInputStream bais = new ByteArrayInputStream(serializedVector);

        final VectorSerializer vectorSerializer = VectorSerializerFactory.getSerializerByStreamContent(bais);

        //testing serialization
        bais.reset();
        final byte[] actualSerializedVector = vectorSerializer.floatToByte(vector);

        assertNotNull(actualSerializedVector);
        assertArrayEquals(serializedVector, actualSerializedVector);

        //testing deserialization
        bais.reset();
        final float[] actualDeserializedVector = vectorSerializer.byteToFloat(bais);

        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);
    }

    public void testVectorAsCollectionOfFloatsSerializer() throws Exception {
        //setup
        int arrayLength = 20;
        float[] vector = new float[arrayLength];
        IntStream.range(0, arrayLength).forEach(index -> vector[index] = random.nextFloat());

        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector)
            ds.writeFloat(f);
        final byte[] vectorAsCollectionOfFloats = bas.toByteArray();
        final ByteArrayInputStream bais = new ByteArrayInputStream(vectorAsCollectionOfFloats);

        final VectorSerializer vectorSerializer = VectorSerializerFactory.getSerializerByStreamContent(bais);

        //testing serialization
        bais.reset();
        final byte[] actualSerializedVector = vectorSerializer.floatToByte(vector);

        assertNotNull(actualSerializedVector);
        assertArrayEquals(vectorAsCollectionOfFloats, actualSerializedVector);

        //testing deserialization
        bais.reset();
        final float[] actualDeserializedVector = vectorSerializer.byteToFloat(bais);

        assertNotNull(actualDeserializedVector);
        assertArrayEquals(vector, actualDeserializedVector, 0.1f);
    }
}
