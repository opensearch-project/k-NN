/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.io.IOException;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.BinaryDocValues;
import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;

import java.io.IOException;

@RequiredArgsConstructor
public final class KNNVectorScriptDocValues extends ScriptDocValues<float[]> {

    private final BinaryDocValues binaryDocValues;
    private final String fieldName;
    @Getter
    private final VectorDataType vectorDataType;
    private boolean docExists = false;

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
            return vectorDataType.getVectorFromBytesRef(binaryDocValues.binaryValue());
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
