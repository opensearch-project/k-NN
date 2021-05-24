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

package org.opensearch.knn.index;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

/**
 * Service to interact with plugin's jni layer. Class dependencies should be minimal
 *
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/index/JNIService.java
 *      src/main/java/org/opensearch/knn/index/KNNQueryResult.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
public class JNIService {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            System.loadLibrary(KNNConstants.JNI_LIBRARY_NAME);

            initLibrary(KNNConstants.NMSLIB_NAME);
            initLibrary(KNNConstants.FAISS_NAME);

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
     * @param engineName name of engine to build index for
     */
    public static native void createIndex(int[] ids, float[][] data, String indexPath, Map<String, Object> parameters,
                                          String engineName);

    /**
     * Create an index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param templateIndex empty template index
     * @param engineName name of engine to build index for
     */
    public static native void createIndexFromTemplate(int[] ids, float[][] data, String indexPath, byte[] templateIndex,
                                                      String engineName);

    /**
     * Load an index into memory
     *
     * @param indexPath path to index file
     * @param parameters parameters to be used when loading index
     * @param engineName name of engine to load index
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndex(String indexPath, Map<String, Object> parameters, String engineName);

    /**
     * Query an index
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param parameters parameters to be used for query
     * @param engineName name of engine to query index
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k,
                                                     Map<String, Object> parameters, String engineName);

    /**
     * Free native memory pointer
     *
     * @param indexPointer location to be freed
     */
    public static native void free(long indexPointer, String engineName);

    /**
     * Initialize library
     *
     * @param engineName name of engine to initialize library for
     */
    public static native void initLibrary(String engineName);

    /**
     * Train an empty index
     *
     * @param indexParameters parameters used to build index
     * @param dimension dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @param engine engine to perform the training
     * @return bytes array of trained template index
     */
    public static native byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer,
                                           String engine);

    /**
     * Transfer vectors from Java to native
     *
     * @param vectorsPointer pointer to vectors in native memory. Should be 0 to create vector as well
     * @param trainingData data to be transferred
     * @return pointer to native memory location of training data
     */
    public static native long transferVectors(long vectorsPointer, float[][] trainingData);

    /**
     * Free vectors from memory
     *
     * @param vectorsPointer to be freed
     */
    public static native void freeVectors(long vectorsPointer);
}
