/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.core.common.Strings;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;

class KNNBuilderUtils {
    public static void validateFieldName(String fieldName, String queryName) {
        if (Strings.isNullOrEmpty(fieldName)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", queryName));
        }
    }

    public static void validateVector(float[] vector, String queryName) {
        if (vector == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", queryName));
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", queryName));
        }
    }

    public static void validateVectorDimension(VectorDataType vectorDataType, int originalVectorLength, int expectedDimension) {
        int vectorLength = VectorDataType.BINARY == vectorDataType ? originalVectorLength * Byte.SIZE : originalVectorLength;
        if (expectedDimension != vectorLength) {
            throw new IllegalArgumentException(
                String.format("Query vector has invalid dimension: %d. Dimension should be: %d", vectorLength, expectedDimension)
            );
        }
    }

    public static MappedFieldType validateAndGetFieldType(String fieldName, QueryShardContext context, boolean ignoreUnmapped) {
        MappedFieldType mappedFieldType = context.fieldMapper(fieldName);
        if (mappedFieldType == null && ignoreUnmapped) {
            return null;
        }
        if (!(mappedFieldType instanceof KNNVectorFieldType)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not knn_vector type.", fieldName));
        }
        return mappedFieldType;
    }

    public static void validateExpandNested(boolean expandNested, BitSetProducer parentFilter) {
        if (parentFilter == null && expandNested) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Invalid value provided for the [%s] field. [%s] is only supported with a nested field.",
                    EXPAND_NESTED,
                    EXPAND_NESTED
                )
            );
        }
    }
}
