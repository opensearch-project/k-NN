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
     * @return parent field path without the child. Null if no parent exists
     */
    public static String getParentField(String field) {
        if (field == null) {
            return null;
        }
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
     * @return child field path without the parent path. Null if no child exists
     */
    public static String getChildField(String field) {
        if (field == null) {
            return null;
        }
        int lastDot = field.lastIndexOf('.');
        if (lastDot == -1) {
            return null;
        }
        return field.substring(lastDot + 1);
    }

    /**
     * Get the child field given a prefix
     *
     * @param field field to extract child from
     * @param prefix prefix to remove up until
     * @return Child field
     */
    public static String getChildField(String field, String prefix) {
        if (field == null) {
            return null;
        }
        if (prefix == null) {
            return getChildField(field);
        }
        if (field.length() <= prefix.length()) {
            return null;
        }

        return field.substring(prefix.length() + 1);
    }

    /**
     * Construct a sibling field path. For instance, if the field is "parent.to.child" and the sibling is "sibling", this
     * would return "parent.to.sibling".
     *
     * @param field   nested field path
     * @param sibling sibling field
     * @return sibling field path
     */
    public static String constructSiblingField(String field, String sibling) {
        String parent = getParentField(field);
        if (parent == null) {
            return sibling;
        }
        return parent + "." + sibling;
    }

    /**
     * Split a nested field path into an array of strings. For instance, if the field is "parent.to.child", this would
     * return ["parent", "to", "child"].
     *
     * @param field nested field path
     * @return array of strings representing the nested field path
     */
    public static String[] splitPath(String field) {
        return field.split("\\.");
    }
}
