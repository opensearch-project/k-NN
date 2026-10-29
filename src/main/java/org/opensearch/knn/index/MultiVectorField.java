/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;

import java.nio.ByteBuffer;
import java.util.List;

public class MultiVectorField extends Field{


    public MultiVectorField(String name, List<float[]> value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            final KNNVectorSerializer vectorSerializer = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE;
            final ByteBuffer floatToByte = vectorSerializer.floatsToByteArray(value);
            this.setBytesValue(floatToByte.array());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param name FieldType name
     * @param value multi arrays of byte vector values
     * @param type FieldType to build DocValues
     */
    public MultiVectorField(String name, byte[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            this.setBytesValue(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
