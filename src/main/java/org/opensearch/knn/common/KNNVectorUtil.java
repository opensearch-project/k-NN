/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Objects;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class KNNVectorUtil {
    /**
     * Check if all the elements of a given vector are zero
     *
     * @param vector the vector
     * @return true if yes; otherwise false
     */
    public static boolean isZeroVector(byte[] vector) {
        Objects.requireNonNull(vector, "vector must not be null");
        for (byte e : vector) {
            if (e != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if all the elements of a given vector are zero
     *
     * @param vector the vector
     * @return true if yes; otherwise false
     */
    public static boolean isZeroVector(float[] vector) {
        Objects.requireNonNull(vector, "vector must not be null");
        for (float e : vector) {
            if (e != 0f) {
                return false;
            }
        }
        return true;
    }

    /**
     * Converts an integer List to and array
     * @param integerList
     * @return null if list is null or empty, int[] otherwise
     */
    public static int[] intListToArray(final List<Integer> integerList) {
        if (integerList == null || integerList.isEmpty()) {
            return null;
        }
        int[] intArray = new int[integerList.size()];
        for (int i = 0; i < integerList.size(); i++) {
            intArray[i] = integerList.get(i);
        }
        return intArray;
    }
}
