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
import org.opensearch.index.mapper.FieldValueFetcher;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * FieldValueFetcher that retrieves KNN vector values from doc values or native KNN vector format.
 *
 * @opensearch.internal
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
            KNNVectorValues<?> vectorValues = fieldInfo != null ? KNNVectorValuesFactory.getVectorValues(fieldInfo, reader) : null;
            if (vectorValues == null) {
                return values;
            }
            if (vectorValues.advance(docId) == docId) {
                values.add(vectorValues.getVector());
            }
        } catch (Exception e) {
            throw new IOException("Failed to read doc values for document " + docId + " in field " + mappedFieldType.name(), e);
        }
        return values;
    }

    /**
     * Converts the raw vector (float[] or byte[]) into a List of Numbers so that
     * XContentBuilder serializes all vector types as numeric JSON arrays.
     * Without this, byte[] would be serialized as Base64 binary data.
     */

    @Override
    public Object convert(Object value) {
        if (value instanceof float[]) {
            float[] vector = (float[]) value;
            List<Number> result = new ArrayList<>(vector.length);
            for (float v : vector) {
                result.add(v);
            }
            return result;
        } else if (value instanceof byte[]) {
            byte[] vector = (byte[]) value;
            List<Number> result = new ArrayList<>(vector.length);
            for (byte v : vector) {
                result.add((int) v);
            }
            return result;
        }
        return value;
    }
}
