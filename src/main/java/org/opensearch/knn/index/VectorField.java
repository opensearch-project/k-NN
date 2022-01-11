/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.VectorSerializer;
import org.opensearch.knn.index.codec.VectorSerializerFactory;

public class VectorField extends Field {

    public VectorField(String name, float[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            final VectorSerializer vectorSerializer = VectorSerializerFactory.getDefaultSerializer();
            final byte[] floatsToBytes = vectorSerializer.floatToByte(value);
            this.setBytesValue(floatsToBytes);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
