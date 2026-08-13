/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.DynamicFieldTypeInferencer;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * k-NN implementation of {@link DynamicFieldTypeInferencer}.
 *
 * <p>Claims unmapped fields whose value is a flat numeric array with at least
 * {@link #MIN_VECTOR_DIMENSION} elements whose count is a multiple of 8, and maps them as
 * {@code knn_vector}. The dimension is inferred from the array length of the first document —
 * subsequent documents with a different dimension are rejected by the mapper.
 *
 * <p>Core hands a factory that produces a fresh parser over the buffered field bytes. We stream
 * the tokens directly rather than materializing a {@code List}: the value must be an array whose
 * every element is a JSON number ({@code VALUE_NUMBER}). That single token check rejects strings,
 * booleans, nulls, and nested arrays/objects (which appear as {@code START_ARRAY}/{@code START_OBJECT}),
 * so a mixed-type array like {@code [1.0, "hello", ...]} falls through to the normal float path
 * instead of being wrongly claimed as a vector.
 *
 * <p>Returns {@code null} for any field that doesn't meet all conditions, allowing subsequent
 * inferencers or the default float fallback to handle it.
 */
public class KNNDynamicFieldTypeInferencer implements DynamicFieldTypeInferencer {

    static final int MIN_VECTOR_DIMENSION = 128;

    /**
     * Streams the buffered field value and returns a knn_vector mapping config if it is a flat
     * numeric array with at least {@link #MIN_VECTOR_DIMENSION} elements whose count is a multiple of 8.
     *
     * @param fieldValueParser produces a fresh parser positioned at the field value's first token
     * @return mutable config map {@code {type: knn_vector, dimension: N}} if claimed, or {@code null} to pass
     */
    @Override
    public Map<String, Object> inferFieldType(FieldValueParserSupplier fieldValueParser) throws IOException {
        int count;
        try (XContentParser parser = fieldValueParser.get()) {
            if (parser.currentToken() != XContentParser.Token.START_ARRAY) {
                return null;
            }
            count = 0;
            XContentParser.Token token;
            while ((token = parser.nextToken()) != XContentParser.Token.END_ARRAY) {
                // Any non-number element (string, boolean, null, nested array/object) disqualifies the field.
                if (token != XContentParser.Token.VALUE_NUMBER) {
                    return null;
                }
                count++;
            }
        }
        if (count < MIN_VECTOR_DIMENSION || count % 8 != 0) {
            return null;
        }
        Map<String, Object> config = new HashMap<>();
        config.put(KNNConstants.TYPE, KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, count);
        return config;
    }
}
