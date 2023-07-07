/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

public final class KNNVectorScriptDocValues extends ScriptDocValues<float[]> {

    private final BinaryDocValues binaryDocValues;
    private final String fieldName;
    @Getter
    private final VectorDataType vectorDataType;
    private boolean docExists;

    public KNNVectorScriptDocValues(BinaryDocValues binaryDocValues, String fieldName, VectorDataType vectorDataType) {
        this.binaryDocValues = binaryDocValues;
        this.fieldName = fieldName;
        this.vectorDataType = vectorDataType;
    }

    @Override
    public void setNextDocId(int docId) throws IOException {
        if (binaryDocValues.advanceExact(docId)) {
            docExists = true;
            return;
        }
        docExists = false;
    }

    public float[] getValue() {
        if (!docExists) {
            String errorMessage = String.format(
                "One of the document doesn't have a value for field '%s'. "
                    + "This can be avoided by checking if a document has a value for the field or not "
                    + "by doc['%s'].size() == 0 ? 0 : {your script}",
                fieldName,
                fieldName
            );
            throw new IllegalStateException(errorMessage);
        }
        try {
            BytesRef value = binaryDocValues.binaryValue();
            if (VectorDataType.BYTE.equals(vectorDataType)) {
                float[] vector = new float[value.length];
                int i = 0;
                int j = value.offset;

                while (i < value.length) {
                    vector[i++] = value.bytes[j++];
                }
                return vector;
            } else if (VectorDataType.FLOAT.equals(vectorDataType)) {
                ByteArrayInputStream byteStream = new ByteArrayInputStream(value.bytes, value.offset, value.length);
                final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                final float[] vector = vectorSerializer.byteToFloatArray(byteStream);
                return vector;
            } else {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s]",
                        VECTOR_DATA_TYPE_FIELD,
                        SUPPORTED_VECTOR_DATA_TYPES
                    )
                );
            }
        } catch (IOException e) {
            throw ExceptionsHelper.convertToOpenSearchException(e);
        }
    }

    @Override
    public int size() {
        return docExists ? 1 : 0;
    }

    @Override
    public float[] get(int i) {
        throw new UnsupportedOperationException("knn vector does not support this operation");
    }
}
