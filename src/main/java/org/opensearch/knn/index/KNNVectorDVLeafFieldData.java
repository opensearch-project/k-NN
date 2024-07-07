/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.index.fielddata.LeafFieldData;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.fielddata.SortedBinaryDocValues;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.search.DocValueFormat;

import java.io.IOException;

public class KNNVectorDVLeafFieldData implements LeafFieldData {

    private final LeafReader reader;
    private final String fieldName;
    private final VectorDataType vectorDataType;

    public KNNVectorDVLeafFieldData(LeafReader reader, String fieldName, VectorDataType vectorDataType) {
        this.reader = reader;
        this.fieldName = fieldName;
        this.vectorDataType = vectorDataType;
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public long ramBytesUsed() {
        return 0; // unknown
    }

    @Override
    public ScriptDocValues<float[]> getScriptValues() {
        try {
            FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(fieldName);
            if (fieldInfo == null) {
                return KNNVectorScriptDocValues.emptyValues(fieldName, vectorDataType);
            }

            DocIdSetIterator values;
            if (fieldInfo.hasVectorValues()) {
                switch (fieldInfo.getVectorEncoding()) {
                    case FLOAT32:
                        values = reader.getFloatVectorValues(fieldName);
                        break;
                    case BYTE:
                        values = reader.getByteVectorValues(fieldName);
                        break;
                    default:
                        throw new IllegalStateException("Unsupported Lucene vector encoding: " + fieldInfo.getVectorEncoding());
                }
            } else {
                values = DocValues.getBinary(reader, fieldName);
            }
            return KNNVectorScriptDocValues.create(values, fieldName, vectorDataType);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load values for knn vector field: " + fieldName, e);
        }
    }

    @Override
    public SortedBinaryDocValues getBytesValues() {
        throw new UnsupportedOperationException("knn vector field '" + fieldName + "' doesn't support sorting");
    }

    @Override
    public DocValueFetcher.Leaf getLeafValueFetcher(DocValueFormat format) {
        final BinaryDocValues binaryDocValues;

        try {
            binaryDocValues = DocValues.getBinary(reader, fieldName);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load KNNDocValues from lucene", e);
        }

        return new DocValueFetcher.Leaf() {
            float[] floats;
            boolean docExists = false;

            @Override
            public boolean advanceExact(int docId) throws IOException {
                if (binaryDocValues.advanceExact(docId)) {
                    docExists = true;
                    floats = vectorDataType.getVectorFromBytesRef(binaryDocValues.binaryValue());
                    return docExists;
                }
                docExists = false;
                return docExists;
            }

            @Override
            public int docValueCount() throws IOException {
                return 1;
            }

            @Override
            public Object nextValue() throws IOException {
                return floats;
            }
        };
    }
}
