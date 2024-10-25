/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

/**
 * Helper class for working with nested fields.
 */
public class ParentChildHelper {

    /**
     * Given a nested field path, return the path of the parent field. For instance if the field is "parent.to.child",
     * this would return "parent.to".
     *
     * @param field nested field path
     * @return parent field path without the child
     */
    public static String getParentField(String field) {
        int lastDot = field.lastIndexOf('.');
        if (lastDot == -1) {
            return null;
        }
        return field.substring(0, lastDot);
    }

    /**
     * Given a nested field path, return the child field. For instance if the field is "parent.to.child", this would
     * return "child".
     *
     * @param field nested field path
     * @return child field path without the parent path
     */
    public static String getChildField(String field) {
        int lastDot = field.lastIndexOf('.');
        return field.substring(lastDot + 1);
    }
}
