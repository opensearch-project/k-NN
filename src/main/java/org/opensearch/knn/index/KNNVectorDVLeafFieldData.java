/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.index.fielddata.LeafFieldData;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.fielddata.SortedBinaryDocValues;
import org.opensearch.knn.common.FieldInfoExtractor;

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
            FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, fieldName);
            if (fieldInfo == null) {
                return KNNVectorScriptDocValues.emptyValues(fieldName, vectorDataType);
            }
            KnnVectorValues knnVectorValues;
            if (fieldInfo.hasVectorValues()) {
                switch (fieldInfo.getVectorEncoding()) {
                    case FLOAT32:
                        knnVectorValues = reader.getFloatVectorValues(fieldName);
                        break;
                    case BYTE:
                        knnVectorValues = reader.getByteVectorValues(fieldName);
                        break;
                    default:
                        throw new IllegalStateException("Unsupported Lucene vector encoding: " + fieldInfo.getVectorEncoding());
                }
                return KNNVectorScriptDocValues.create(knnVectorValues, fieldName, vectorDataType);
            }
            DocIdSetIterator values = DocValues.getBinary(reader, fieldName);
            return KNNVectorScriptDocValues.create(values, fieldName, vectorDataType);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load values for knn vector field: " + fieldName, e);
        }
    }

    @Override
    public SortedBinaryDocValues getBytesValues() {
        throw new UnsupportedOperationException("knn vector field '" + fieldName + "' doesn't support sorting");
    }
}
