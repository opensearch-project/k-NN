/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.io.IOException;
import java.util.Objects;
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
public abstract class KNNVectorScriptDocValues<T> extends ScriptDocValues<T> {

    private final DocIdSetIterator vectorValues;
    private final String fieldName;
    @Getter
    private final VectorDataType vectorDataType;
    private boolean docExists = false;
    private int lastDocID = -1;

    @Override
    public void setNextDocId(int docId) throws IOException {
        if (docId < lastDocID) {
            throw new IllegalArgumentException("docs were sent out-of-order: lastDocID=" + lastDocID + " vs docID=" + docId);
        }

        lastDocID = docId;

        int curDocID = vectorValues.docID();
        if (lastDocID > curDocID) {
            curDocID = vectorValues.advance(docId);
        }
        docExists = lastDocID == curDocID;
    }

    public T getValue() {
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

    protected abstract T doGetValue() throws IOException;

    @Override
    public int size() {
        return docExists ? 1 : 0;
    }

    @Override
    public T get(int i) {
        throw new UnsupportedOperationException("knn vector does not support this operation");
    }

    /**
     * Creates a KNNVectorScriptDocValues object based on the provided parameters.
     *
     * @param values          The DocIdSetIterator representing the vector values.
     * @param fieldName       The name of the field.
     * @param vectorDataType  The data type of the vector.
     * @return A KNNVectorScriptDocValues object based on the type of the values.
     * @throws IllegalArgumentException If the type of values is unsupported.
     */
    public static KNNVectorScriptDocValues<?> create(DocIdSetIterator values, String fieldName, VectorDataType vectorDataType) {
        Objects.requireNonNull(values, "values must not be null");
        if (values instanceof ByteVectorValues) {
            return new KNNByteVectorScriptDocValues((ByteVectorValues) values, fieldName, vectorDataType);
        } else if (values instanceof FloatVectorValues) {
            return new KNNFloatVectorScriptDocValues((FloatVectorValues) values, fieldName, vectorDataType);
        } else if (values instanceof BinaryDocValues) {
            return new KNNNativeVectorScriptDocValues<>((BinaryDocValues) values, fieldName, vectorDataType);
        } else {
            throw new IllegalArgumentException("Unsupported values type: " + values.getClass());
        }
    }

    private static final class KNNByteVectorScriptDocValues extends KNNVectorScriptDocValues<byte[]> {
        private final ByteVectorValues values;

        KNNByteVectorScriptDocValues(ByteVectorValues values, String field, VectorDataType type) {
            super(values, field, type);
            this.values = values;
        }

        @Override
        protected byte[] doGetValue() throws IOException {
            try {
                return values.vectorValue();
            } catch (IOException e) {
                throw ExceptionsHelper.convertToOpenSearchException(e);
            }
        }
    }

    private static final class KNNFloatVectorScriptDocValues extends KNNVectorScriptDocValues<float[]> {
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

    private static final class KNNNativeVectorScriptDocValues<T> extends KNNVectorScriptDocValues<T> {
        private final BinaryDocValues values;

        KNNNativeVectorScriptDocValues(BinaryDocValues values, String field, VectorDataType type) {
            super(values, field, type);
            this.values = values;
        }

        @Override
        protected T doGetValue() throws IOException {
            return getVectorDataType().getVectorFromBytesRef(values.binaryValue());
        }
    }

    /**
     * Creates an empty KNNVectorScriptDocValues object based on the provided field name and vector data type.
     *
     * @param fieldName The name of the field.
     * @param type      The data type of the vector.
     * @return An empty KNNVectorScriptDocValues object.
     */
    public static KNNVectorScriptDocValues<?> emptyValues(String fieldName, VectorDataType type) {
        if (type == VectorDataType.FLOAT) {
            return new KNNVectorScriptDocValues<float[]>(DocIdSetIterator.empty(), fieldName, type) {
                @Override
                protected float[] doGetValue() throws IOException {
                    throw new UnsupportedOperationException("empty values");
                }
            };
        }
        return new KNNVectorScriptDocValues<byte[]>(DocIdSetIterator.empty(), fieldName, type) {
            @Override
            protected byte[] doGetValue() throws IOException {
                throw new UnsupportedOperationException("empty values");
            }
        };
    }
}
