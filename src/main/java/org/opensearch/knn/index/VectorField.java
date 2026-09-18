/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;

public class VectorField extends Field {

    public VectorField(String name, float[] value, IndexableFieldType type) {
        this(name, value, type, VectorDataType.FLOAT);
    }

    /**
     * Serializes a float vector into the binary DocValues representation of the given data type. HALF_FLOAT is
     * encoded as FP16 (2 bytes per dimension), everything else as FP32 (4 bytes per dimension). The encoding must
     * match {@link VectorDataType#getVectorFromBytesRef(BytesRef)}, which is what reads these bytes back.
     *
     * @param name FieldType name
     * @param value an array of float vector values
     * @param type FieldType to build DocValues
     * @param vectorDataType data type the vector is stored as
     */
    public VectorField(String name, float[] value, IndexableFieldType type, VectorDataType vectorDataType) {
        super(name, new BytesRef(), type);
        try {
            this.setBytesValue(serialize(value, vectorDataType));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param name FieldType name
     * @param value an array of byte vector values
     * @param type FieldType to build DocValues
     */
    public VectorField(String name, byte[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            this.setBytesValue(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    private static byte[] serialize(final float[] value, final VectorDataType vectorDataType) {
        if (VectorDataType.HALF_FLOAT == vectorDataType) {
            final byte[] halfFloatToByte = new byte[value.length * 2];
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(value, halfFloatToByte, value.length);
            return halfFloatToByte;
        }
        final KNNVectorSerializer vectorSerializer = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE;
        return vectorSerializer.floatToByteArray(value);
    }
}
