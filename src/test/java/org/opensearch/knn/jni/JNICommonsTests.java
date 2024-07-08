/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.KNNTestCase;

public class JNICommonsTests extends KNNTestCase {

    public void testStoreVectorData_whenVaildInputThenSuccess() {
        float[][] data = new float[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                data[i][j] = i + j;
            }
        }
        long memoryAddress = JNICommons.storeVectorData(0, data, 8);
        assertTrue(memoryAddress > 0);
        assertEquals(memoryAddress, JNICommons.storeVectorData(memoryAddress, data, 8));
    }

    public void testFreeVectorData_whenValidInput_ThenSuccess() {
        float[][] data = new float[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                data[i][j] = i + j;
            }
        }
        long memoryAddress = JNICommons.storeVectorData(0, data, 8);
        JNICommons.freeVectorData(memoryAddress);
    }
}
