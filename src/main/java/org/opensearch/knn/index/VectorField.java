/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;

public class VectorField extends Field {

    public VectorField(String name, float[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            this.setBytesValue(floatToByte(value));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static byte[] floatToByte(float[] floats) throws Exception {
        byte[] bytes;
        try (ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
             ObjectOutputStream objectStream = new ObjectOutputStream(byteStream);) {
            objectStream.writeObject(floats);
            bytes = byteStream.toByteArray();
        }
        return bytes;
    }
}
