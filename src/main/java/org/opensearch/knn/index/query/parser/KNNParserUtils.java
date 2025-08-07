/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.XContentLocation;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

class KNNParserUtils {
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
