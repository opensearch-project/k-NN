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
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import static org.opensearch.knn.index.KNNSettings.isFaissAVX2Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512SPRDisabled;
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SPRSupportedBySystem;

/**
 * Service to interact with faiss jni layer. Class dependencies should be minimal
 * <p>
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/FaissService.java
 *      src/main/java/org/opensearch/knn/index/query/KNNQueryResult.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
class FaissService {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {

            // Even if the underlying system supports AVX512 and AVX2, users can override and disable it by setting
            // 'knn.faiss.avx2.disabled', 'knn.faiss.avx512.disabled', or 'knn.faiss.avx512_spr.disabled' to true in the opensearch.yml
            // configuration
            if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
                System.loadLibrary(KNNConstants.FAISS_AVX512_SPR_JNI_LIBRARY_NAME);
            } else if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
                System.loadLibrary(KNNConstants.FAISS_AVX512_JNI_LIBRARY_NAME);
            } else if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
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
     * Initialize an index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs number of documents to be added
     * @param dim dimension of the vector to be indexed
     * @param parameters parameters to build index
     */
    public static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    /**
     * Initialize an index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs number of documents to be added
     * @param dim dimension of the vector to be indexed
     * @param parameters parameters to build index
     */
    public static native long initBinaryIndex(long numDocs, int dim, Map<String, Object> parameters);

    /**
     * Initialize a byte index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs number of documents to be added
     * @param dim dimension of the vector to be indexed
     * @param parameters parameters to build index
     */
    public static native long initByteIndex(long numDocs, int dim, Map<String, Object> parameters);

    /**
     * Inserts to a faiss index. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory.
     *
     * @param ids ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim dimension of the vector to be indexed
     * @param indexAddress address of native memory where index is stored
     * @param threadCount number of threads to use for insertion
     */
    public static native void insertToIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    /**
     * Inserts to a faiss index. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory.
     *
     * @param ids ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim dimension of the vector to be indexed
     * @param indexAddress address of native memory where index is stored
     * @param threadCount number of threads to use for insertion
     */
    public static native void insertToBinaryIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    /**
     * Inserts to a faiss index. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory.
     *
     * @param ids ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim dimension of the vector to be indexed
     * @param indexAddress address of native memory where index is stored
     * @param threadCount number of threads to use for insertion
     */
    public static native void insertToByteIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    /**
     * Writes a faiss index.
     *
     * NOTE: This will always free the index. Do not call free after this.
     *
     * @param indexAddress address of native memory where index is stored
     * @param output Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     */
    public static native void writeIndex(long indexAddress, IndexOutputWithBuffer output);

    /**
     * Writes a faiss index.
     *
     * NOTE: This will always free the index. Do not call free after this.
     *
     * @param indexAddress address of native memory where index is stored
     * @param output Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     */
    public static native void writeBinaryIndex(long indexAddress, IndexOutputWithBuffer output);

    /**
     * Writes a faiss index.
     *
     * NOTE: This will always free the index. Do not call free after this.
     *
     * @param indexAddress address of native memory where index is stored
     * @param output Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     */
    public static native void writeByteIndex(long indexAddress, IndexOutputWithBuffer output);

    /**
     * Create an index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim dimension of the vector to be indexed
     * @param output Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex empty template index
     * @param parameters additional build time parameters
     */
    public static native void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    /**
     * Create a binary index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim dimension of the vector to be indexed
     * @param output Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex empty template index
     * @param parameters additional build time parameters
     */
    public static native void createBinaryIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    /**
     * Create a byte index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim dimension of the vector to be indexed
     * @param output Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex empty template index
     * @param parameters additional build time parameters
     */
    public static native void createByteIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
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
     * Load an index into memory via a wrapping having Lucene's IndexInput.
     * Instead of directly accessing an index path, this will make Faiss delegate IndexInput to load bytes.
     *
     * @param readStream IndexInput wrapper having a Lucene's IndexInput reference.
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndexWithStream(IndexInputWithBuffer readStream);

    /**
     * Load a binary index into memory
     *
     * @param indexPath path to index file
     * @return pointer to location in memory the index resides in
     */
    public static native long loadBinaryIndex(String indexPath);

    /**
     * Load a binary index into memory with a wrapping having Lucene's IndexInput.
     * Instead of directly accessing an index path, this will make Faiss delegate IndexInput to load bytes.
     *
     * @param readStream IndexInput wrapper having a Lucene's IndexInput reference.
     * @return pointer to location in memory the index resides in
     */
    public static native long loadBinaryIndexWithStream(IndexInputWithBuffer readStream);

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
     * @param methodParameters method parameter
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        int[] parentIds
    );

    /**
     * Query an index with filter
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param methodParameters method parameter
     * @param filterIds list of doc ids to include in the query result
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndexWithFilter(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filterIds,
        int filterIdsType,
        int[] parentIds
    );

    /**
     * Query a binary index with filter
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param methodParameters method parameter
     * @param filterIds list of doc ids to include in the query result
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryBinaryIndexWithFilter(
        long indexPointer,
        byte[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filterIds,
        int filterIdsType,
        int[] parentIds
    );

    /**
     * Query a binary index with filter
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param filterIds list of doc ids to include in the query result
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryBinaryIndexWithFilter(
        long indexPointer,
        byte[] queryVector,
        int k,
        long[] filterIds,
        int filterIdsType,
        int[] parentIds
    );

    /**
     * Free native memory pointer
     */
    public static native void free(long indexPointer, boolean isBinary);

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
     * Train an empty binary index
     *
     * @param indexParameters parameters used to build index
     * @param dimension dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainBinaryIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Train an empty byte index
     *
     * @param indexParameters parameters used to build index
     * @param dimension dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainByteIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Range search index with filter
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param radius search within radius threshold
     * @param methodParameters parameters to be used for the query
     * @param indexMaxResultWindow maximum number of results to return
     * @param filteredIds list of doc ids to include in the query result
     * @param filterIdsType type of filter ids
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of neighbors within radius
     */
    public static native KNNQueryResult[] rangeSearchIndexWithFilter(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    /**
     * Range search index
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param radius search within radius threshold
     * @param methodParameters parameters to be used for the query
     * @param indexMaxResultWindow maximum number of results to return
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of neighbors within radius
     */
    public static native KNNQueryResult[] rangeSearchIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        int[] parentIds
    );
}
