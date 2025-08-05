/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.Strings;
import org.opensearch.core.xcontent.XContentLocation;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;

public class KNNBuilderAndParserUtils {
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

    public static void validateVectorDimension(int vectorLength, int expectedDimension) {
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

    public static float[] floatListToFloatArray(List<Float> floats, String queryName) {
        if (Objects.isNull(floats) || floats.isEmpty()) {
            throw new IllegalArgumentException(String.format("[%s] field 'vector' requires to be non-null and non-empty", queryName));
        }
        float[] vec = new float[floats.size()];
        for (int i = 0; i < floats.size(); i++) {
            vec[i] = floats.get(i);
        }
        return vec;
    }

    public static float[] objectsToFloats(List<Object> objs, String queryName) {
        if (Objects.isNull(objs) || objs.isEmpty()) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "[%s] field 'vector' requires to be non-null and non-empty", queryName)
            );
        }
        float[] vec = new float[objs.size()];
        for (int i = 0; i < objs.size(); i++) {
            if ((objs.get(i) instanceof Number) == false) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] field 'vector' requires to be an array of numbers", queryName)
                );
            }
            vec[i] = ((Number) objs.get(i)).floatValue();
        }
        return vec;
    }

    public static void throwParsingExceptionOnMultipleFields(
        XContentLocation contentLocation,
        String processedFieldName,
        String currentFieldName,
        String queryName
    ) {
        if (processedFieldName != null) {
            throw new ParsingException(
                contentLocation,
                "["
                    + queryName
                    + "] query doesn't support multiple fields, found ["
                    + processedFieldName
                    + "] and ["
                    + currentFieldName
                    + "]"
            );
        }
    }
}
