/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.FieldValueFetcher;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * FieldValueFetcher that retrieves KNN vector values from doc values or native KNN vector format.
 *
 */
public class KnnVectorValuesFetcher extends FieldValueFetcher {
    KNNVectorFieldType mappedFieldType;

    public KnnVectorValuesFetcher(KNNVectorFieldType mappedFieldType, String simpleName) {
        super(simpleName);
        this.mappedFieldType = mappedFieldType;
    }

    @Override
    public List<Object> fetch(LeafReader reader, int docId) throws IOException {
        List<Object> values = new ArrayList<>(1);
        try {
            FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(mappedFieldType.name());
            KNNVectorValues<?> vectorValues = fieldInfo != null
                ? KNNVectorValuesFactory.getVectorValues(fieldInfo, Lucene.segmentReader(reader))
                : null;
            if (vectorValues == null) {
                return values;
            }
            if (vectorValues.advance(docId) == docId) {
                values.add(vectorValues.getVector());
            }
        } catch (Exception e) {
            throw new IOException("Failed to read vector values for document " + docId + " in field " + mappedFieldType.name(), e);
        }
        return values;
    }

    /**
     * Converts byte[] vectors into int[] so that XContentBuilder serializes them as numeric JSON
     * arrays instead of Base64 binary data. float[] needs no conversion — XContentBuilder natively
     * handles float[] via field(String, float[]).
     * Uses same approach as {@link org.opensearch.knn.index.codec.derivedsource.DerivedSourceIndexOperationListener#formatVector}.
     */
    @Override
    public Object convert(Object value) {
        if (value instanceof byte[] vector) {
            return KNNVectorFieldMapperUtil.deserializeStoredVector(new BytesRef(vector), mappedFieldType.getVectorDataType());
        }
        return value;
    }
}
