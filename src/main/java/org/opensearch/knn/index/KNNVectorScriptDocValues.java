/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.knn.index.codec.VectorSerializer;
import org.opensearch.knn.index.codec.VectorSerializerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public final class KNNVectorScriptDocValues extends ScriptDocValues<float[]> {

    private final BinaryDocValues binaryDocValues;
    private final String fieldName;
    private boolean docExists;

    public KNNVectorScriptDocValues(BinaryDocValues binaryDocValues, String fieldName) {
        this.binaryDocValues = binaryDocValues;
        this.fieldName = fieldName;
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
                "One of the document doesn't have a value for field '%s'. " +
                "This can be avoided by checking if a document has a value for the field or not " +
                "by doc['%s'].size() == 0 ? 0 : {your script}",fieldName,fieldName);
            throw new IllegalStateException(errorMessage);
        }
        try {
            BytesRef value = binaryDocValues.binaryValue();
            ByteArrayInputStream byteStream = new ByteArrayInputStream(value.bytes, value.offset, value.length);
            final VectorSerializer vectorSerializer = VectorSerializerFactory.getSerializerByStreamContent(byteStream);
            final float[] vector = vectorSerializer.byteToFloat(byteStream);
            return vector;
        } catch (IOException e) {
            throw ExceptionsHelper.convertToOpenSearchException(e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException((e));
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
