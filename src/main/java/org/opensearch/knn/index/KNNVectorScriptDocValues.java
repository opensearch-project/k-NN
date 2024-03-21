/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.io.IOException;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;

@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public abstract class KNNVectorScriptDocValues extends ScriptDocValues<float[]> {

    private final DocIdSetIterator vectorValues;
    private final String fieldName;
    @Getter
    private final VectorDataType vectorDataType;
    private boolean docExists = false;

    @Override
    public void setNextDocId(int docId) throws IOException {
        docExists = vectorValues.docID() == docId || vectorValues.advance(docId) == docId;
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
            return doGetValue();
        } catch (IOException e) {
            throw ExceptionsHelper.convertToOpenSearchException(e);
        }
    }

    protected abstract float[] doGetValue() throws IOException;

    @Override
    public int size() {
        return docExists ? 1 : 0;
    }

    @Override
    public float[] get(int i) {
        throw new UnsupportedOperationException("knn vector does not support this operation");
    }

    public static KNNVectorScriptDocValues create(DocIdSetIterator values, String fieldName, VectorDataType vectorDataType) {
        if (values instanceof ByteVectorValues) {
            return new KNNByteVectorScriptDocValues((ByteVectorValues) values, fieldName, vectorDataType);
        } else if (values instanceof FloatVectorValues) {
            return new KNNFloatVectorScriptDocValues((FloatVectorValues) values, fieldName, vectorDataType);
        } else if (values instanceof BinaryDocValues) {
            return new KNNNativeVectorScriptDocValues((BinaryDocValues) values, fieldName, vectorDataType);
        } else {
            throw new IllegalArgumentException("Unsupported values type: " + values.getClass());
        }
    }

    private static final class KNNByteVectorScriptDocValues extends KNNVectorScriptDocValues {
        private final ByteVectorValues values;

        KNNByteVectorScriptDocValues(ByteVectorValues values, String field, VectorDataType type) {
            super(values, field, type);
            this.values = values;
        }

        @Override
        protected float[] doGetValue() throws IOException {
            byte[] bytes = values.vectorValue();
            float[] value = new float[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                value[i] = (float) bytes[i];
            }
            return value;
        }
    }

    private static final class KNNFloatVectorScriptDocValues extends KNNVectorScriptDocValues {
        private final FloatVectorValues values;

        KNNFloatVectorScriptDocValues(FloatVectorValues values, String field, VectorDataType type) {
            super(values, field, type);
            this.values = values;
        }

        @Override
        protected float[] doGetValue() throws IOException {
            return values.vectorValue();
        }
    }

    private static final class KNNNativeVectorScriptDocValues extends KNNVectorScriptDocValues {
        private final BinaryDocValues values;

        KNNNativeVectorScriptDocValues(BinaryDocValues values, String field, VectorDataType type) {
            super(values, field, type);
            this.values = values;
        }

        @Override
        protected float[] doGetValue() throws IOException {
            return getVectorDataType().getVectorFromDocValues(values.binaryValue());
        }
    }
}
