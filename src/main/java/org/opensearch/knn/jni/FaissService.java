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
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.util.KNNEngine;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import static org.opensearch.knn.index.KNNSettings.isFaissAVX2Disabled;
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;

/**
 * Service to interact with faiss jni layer. Class dependencies should be minimal
 *
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/FaissService.java
 *      src/main/java/org/opensearch/knn/index/query/KNNQueryResult.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
class FaissService {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {

            // Even if the underlying system supports AVX2, users can override and disable it by using the
            // 'knn.faiss.avx2.disabled' setting by setting it to true in the opensearch.yml configuration
            if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
                System.loadLibrary(KNNConstants.FAISS_AVX2_JNI_LIBRARY_NAME);
            } else {
                System.loadLibrary(KNNConstants.FAISS_JNI_LIBRARY_NAME);
            }

            initLibrary();
            KNNEngine.FAISS.setInitialized(true);
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
     * Create an index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param templateIndex empty template index
     * @param parameters additional build time parameters
     */
    public static native void createIndexFromTemplate(
        int[] ids,
        float[][] data,
        String indexPath,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    /**
     * Load an index into memory
     *
     * @param indexPath path to index file
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndex(String indexPath);

    /**
     * Determine if index contains shared state.
     *
     * @param indexAddr address of index to be checked.
     * @return true if index requires shared index state; false otherwise
     */
    public static native boolean isSharedIndexStateRequired(long indexAddr);

    /**
     * Initialize the shared state for an index
     *
     * @param indexAddr address of the index to initialize from
     * @return Address of shared index state address
     */
    public static native long initSharedIndexState(long indexAddr);

    /**
     * Set the index state for an index
     *
     * @param indexAddr address of index to set state for
     * @param shareIndexStateAddr address of shared state to be set
     */
    public static native void setSharedIndexState(long indexAddr, long shareIndexStateAddr);

    /**
     * Query an index without filter
     *
     * If the "knn" field is a nested field, each vector value within that nested field will be assigned its
     * own document ID. In this situation, the term "parent ID" corresponds to the original document ID.
     * The arrangement of parent IDs and nested field IDs is assured to have all nested field IDs appearing first,
     * followed by the parent ID, in consecutive order without any gaps. Because of this ID pattern,
     * we can determine the parent ID of a specific nested field ID using only an array of parent IDs.
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k, int[] parentIds);

    /**
     * Query an index with filter
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param filterIds list of doc ids to include in the query result
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndexWithFilter(
        long indexPointer,
        float[] queryVector,
        int k,
        long[] filterIds,
        int filterIdsType,
        int[] parentIds
    );

    /**
     * Free native memory pointer
     */
    public static native void free(long indexPointer);

    /**
     * Deallocate memory of the shared index state
     *
     * @param shareIndexStateAddr address of shared state
     */
    public static native void freeSharedIndexState(long shareIndexStateAddr);

    /**
     * Initialize library
     *
     */
    public static native void initLibrary();

    /**
     * Train an empty index
     *
     * @param indexParameters parameters used to build index
     * @param dimension dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

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

    /**
     * Range search index
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param radius search within radius threshold
     * @param indexMaxResultWindow maximum number of results to return
     * @return KNNQueryResult array of neighbors within radius
     */
    public static native KNNQueryResult[] rangeSearchIndex(long indexPointer, float[] queryVector, float radius, int indexMaxResultWindow);
}
