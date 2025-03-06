/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.extern.log4j.Log4j2;
import org.opensearch.common.xcontent.support.XContentMapValues;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility class for manipulating the source map
 */
@Log4j2
public class DerivedSourceMapHelper {

    /**
     * Removes all fields in the array from the source
     *
     * @param fields Fields to remove
     * @param source Source map to remove from
     * @return Map with filtered fields
     */
    public static Map<String, Object> filterFields(String[] fields, Map<String, Object> source) {
        return XContentMapValues.filter(null, fields).apply(source);
    }

    /**
     * Check whether field exists in the document
     *
     * @param source source document
     * @param fieldName field to check. Field should be flattened. i.e. my.path.field
     * @return whether or not the field exists in the object
     */
    public static boolean fieldExists(Map<String, Object> source, String fieldName) {
        return XContentMapValues.extractValue(fieldName, source, NullValue.INSTANCE) != null;
    }

    /**
     * Injects vector into source, handling object field path if necessary
     *
     * @param sourceAsMap source to be injected into
     * @param vector vector to inject
     * @param fieldName field name injecting at
     */
    public static void injectVector(Map<String, Object> sourceAsMap, Object vector, String fieldName) {
        // If a field contains ".", we need to ensure that we properly nest it.
        String[] fields = ParentChildHelper.splitPath(fieldName);
        if (fields.length < 2) {
            sourceAsMap.put(fieldName, vector);
        }

        Map<String, Object> currentMap = sourceAsMap;
        for (int i = 0; i < fields.length - 1; i++) {
            String field = fields[i];
            currentMap = (Map<String, Object>) currentMap.computeIfAbsent(field, k -> new HashMap<>());
        }
        currentMap.put(fields[fields.length - 1], vector);

    }

    private static class NullValue {
        private static final NullValue INSTANCE = new NullValue();
    }
}
