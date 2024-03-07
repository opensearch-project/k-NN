/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

public class KNNVectorUtil {
    private KNNVectorUtil() {}

    /**
     * Check if all the elements of a given vector are zero
     *
     * @param vector the vector
     * @return true if yes; otherwise false
     */
    public static boolean isZeroVector(byte[] vector) {
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
        for (float e : vector) {
            if (e != 0f) {
                return false;
            }
        }
        return true;
    }
}
