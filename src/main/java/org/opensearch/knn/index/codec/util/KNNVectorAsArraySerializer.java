/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;

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
    public byte[] floatToByteArray(float[] input) {
        byte[] bytes;
        try (
            ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
            ObjectOutputStream objectStream = new ObjectOutputStream(byteStream);
        ) {
            objectStream.writeObject(input);
            bytes = byteStream.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return bytes;
    }

    @Override
    public float[] byteToFloatArray(BytesRef bytesRef) {
        try (ByteArrayInputStream byteStream = new ByteArrayInputStream(bytesRef.bytes, bytesRef.offset, bytesRef.length)) {
            final ObjectInputStream objectStream = new ObjectInputStream(byteStream);
            final float[] vector = (float[]) objectStream.readObject();
            return vector;
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
