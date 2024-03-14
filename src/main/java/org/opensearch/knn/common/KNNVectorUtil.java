/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import java.util.Objects;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

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
}
