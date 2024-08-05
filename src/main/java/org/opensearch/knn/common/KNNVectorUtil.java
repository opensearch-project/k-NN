/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
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
     * Creates an int overflow safe arraylist. If there is an overflow it will create a list with default initial size
     *
     * Default size of array list is 10 and increases 1.5x. The method assigns a contiguous block of memory when the
     * user knows the memory needed. This avoids continuous internal resizing of arraylist
     *
     * @param initialCapacity size to allocate
     * @return an arrayList of size initialCapacity or default size
     */
    public static <T> ArrayList<T> intOverflowSafeArrayList(long initialCapacity) {
        try {
            return new ArrayList<>(Math.toIntExact(initialCapacity));
        } catch (Exception exception) {
            // No-op
        }
        return new ArrayList<>();
    }
}
