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

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.CustomKNNQueryResult;
import org.opensearch.knn.index.util.KNNEngine;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

/**
 * Service to interact with nmslib jni layer. Class dependencies should be minimal
 *
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/NmslibService.java
 *      src/main/java/org/opensearch/knn/index/query/CustomKNNQueryResult.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 *
 * TODO: This needs to be fixed. Currently, in order to get the command to compile, we need to comment out the
 * KNNEngine related code. We need to remove the dependency of this class on KNNEngine. Related issue:
 * <a href="https://github.com/opensearch-project/k-NN/issues/453">...</a>
 */
class NmslibService {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            System.loadLibrary(KNNConstants.NMSLIB_JNI_LIBRARY_NAME);
            initLibrary();
            KNNEngine.NMSLIB.setInitialized(true);
            return null;
        });
    }

    /**
     * Create an index for the native library
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param parameters parameters to build index
     */
    public static native void createIndex(int[] ids, float[][] data, String indexPath, Map<String, Object> parameters);

    /**
     * Load an index into memory
     *
     * @param indexPath path to index file
     * @param parameters parameters to be used when loading index
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndex(String indexPath, Map<String, Object> parameters);

    /**
     * Query an index
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @return KNNQueryResult array of k neighbors
     */
    public static native CustomKNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k);

    /**
     * Free native memory pointer
     */
    public static native void free(long indexPointer);

    /**
     * Initialize library
     */
    public static native void initLibrary();

}
