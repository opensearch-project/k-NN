/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

public class SimdVectorComputeService {
    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            System.loadLibrary(KNNConstants.DEFAULT_SIMD_COMPUTING_JNI_LIBRARY_NAME);
            return null;
        });
    }

    public enum SimilarityFunctionType {
        FP16_MAXIMUM_INNER_PRODUCT,
        FP16_L2,
    }

    public native static void bulkDistanceCalculation(int[] internalVectorIds, float[] scores, int numVectors);

    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    public native static float scoreSingleVector(int internalVectorId);
}
