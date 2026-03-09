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

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.MergeAbortChecker;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.util.Map;

/**
 * Service to interact with faiss jni layer. Class dependencies should be minimal
 * <p>
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/FaissService.java
 * src/main/java/org/opensearch/knn/index/query/KNNQueryResult.java
 * src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
@Log4j2
class FaissService {

    static {
        KNNLibraryLoader.loadFaissLibrary();
        initLibrary();
        KNNEngine.FAISS.setInitialized(true);

        try {
            MergeAbortChecker.isMergeAborted();
            setMergeInterruptCallback();
        } catch (Exception e) {
            // Ignore merge abort callback
            log.warn("Unable to add the mergeAbortChecker during Faiss Initialization", e);
        }
    }

    /**
     * Initialize an index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs    number of documents to be added
     * @param dim        dimension of the vector to be indexed
     * @param parameters parameters to build index
     */
    public static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    /**
     * Initialize an index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs    number of documents to be added
     * @param dim        dimension of the vector to be indexed
     * @param parameters parameters to build index
     */
    public static native long initBinaryIndex(long numDocs, int dim, Map<String, Object> parameters);

    /**
     * Initialize a byte index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs    number of documents to be added
     * @param dim        dimension of the vector to be indexed
     * @param parameters parameters to build index
     */
    public static native long initByteIndex(long numDocs, int dim, Map<String, Object> parameters);

    /**
     * Inserts to a faiss index. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory.
     *
     * @param ids            ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param indexAddress   address of native memory where index is stored
     * @param threadCount    number of threads to use for insertion
     */
    public static native void insertToIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    /**
     * Inserts to a faiss index. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory.
     *
     * @param ids            ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param indexAddress   address of native memory where index is stored
     * @param threadCount    number of threads to use for insertion
     */
    public static native void insertToBinaryIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    /**
     * Inserts to a faiss index. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory.
     *
     * @param ids            ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param indexAddress   address of native memory where index is stored
     * @param threadCount    number of threads to use for insertion
     */
    public static native void insertToByteIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    /**
     * Writes a faiss index.
     *
     * NOTE: This will always free the index. Do not call free after this.
     *
     * @param indexAddress address of native memory where index is stored
     * @param output       Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     */
    public static native void writeIndex(long indexAddress, IndexOutputWithBuffer output);

    /**
     * Writes a faiss index.
     *
     * NOTE: This will always free the index. Do not call free after this.
     *
     * @param indexAddress address of native memory where index is stored
     * @param output       Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     */
    public static native void writeBinaryIndex(long indexAddress, IndexOutputWithBuffer output);

    /**
     * Writes a faiss index.
     *
     * NOTE: This will always free the index. Do not call free after this.
     *
     * @param indexAddress address of native memory where index is stored
     * @param output       Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     */
    public static native void writeByteIndex(long indexAddress, IndexOutputWithBuffer output);

    /**
     * Create an index for the native library with a provided template index
     *
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param output         Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex  empty template index
     * @param parameters     additional build time parameters
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
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param output         Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex  empty template index
     * @param parameters     additional build time parameters
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
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param output         Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex  empty template index
     * @param parameters     additional build time parameters
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
     * Load an index into memory via a wrapping having Lucene's IndexInput with ADC
     *
     * @param readStream IndexInput wrapper having a Lucene's IndexInput reference.
     * @param parameters Map<String, Object> containing the following:
     *                   SpaceType: l2 or innerproduct
     *                   quantizationlevel: Based on the ScalarQuantizationParams type identifier passed in.
     *                   Currently only ScalarQuantizationParams_1 is supported for one-bit ADC
     *                   (@see ScalarQuantizationParams#generateTypeIdentifier)
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndexWithStreamADCParams(IndexInputWithBuffer readStream, Map<String, Object> parameters);

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
     * @param indexAddr           address of index to set state for
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
     * @param indexPointer     pointer to index in memory
     * @param queryVector      vector to be used for query
     * @param k                neighbors to be returned
     * @param methodParameters method parameter
     * @param parentIds        list of parent doc ids when the knn field is a nested field
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
     * @param indexPointer     pointer to index in memory
     * @param queryVector      vector to be used for query
     * @param k                neighbors to be returned
     * @param methodParameters method parameter
     * @param filterIds        list of doc ids to include in the query result
     * @param parentIds        list of parent doc ids when the knn field is a nested field
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
     * @param indexPointer     pointer to index in memory
     * @param queryVector      vector to be used for query
     * @param k                neighbors to be returned
     * @param methodParameters method parameter
     * @param filterIds        list of doc ids to include in the query result
     * @param parentIds        list of parent doc ids when the knn field is a nested field
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
     * @param queryVector  vector to be used for query
     * @param k            neighbors to be returned
     * @param filterIds    list of doc ids to include in the query result
     * @param parentIds    list of parent doc ids when the knn field is a nested field
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
     * @param indexParameters     parameters used to build index
     * @param dimension           dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Train an empty binary index
     *
     * @param indexParameters     parameters used to build index
     * @param dimension           dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainBinaryIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Train an empty byte index
     *
     * @param indexParameters     parameters used to build index
     * @param dimension           dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainByteIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Range search index with filter
     *
     * @param indexPointer         pointer to index in memory
     * @param queryVector          vector to be used for query
     * @param radius               search within radius threshold
     * @param methodParameters     parameters to be used for the query
     * @param indexMaxResultWindow maximum number of results to return
     * @param filteredIds          list of doc ids to include in the query result
     * @param filterIdsType        type of filter ids
     * @param parentIds            list of parent doc ids when the knn field is a nested field
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
     * @param indexPointer         pointer to index in memory
     * @param queryVector          vector to be used for query
     * @param radius               search within radius threshold
     * @param methodParameters     parameters to be used for the query
     * @param indexMaxResultWindow maximum number of results to return
     * @param parentIds            list of parent doc ids when the knn field is a nested field
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

    /**
     * Sets the merge interrupt callback for Faiss operations.
     *
     * <p>This method initializes a singleton interrupt callback that allows Faiss operations
     * to be aborted when Lucene merge operations are cancelled. The callback uses
     * {@link org.apache.lucene.index.MergeAbortChecker} to detect when the current thread
     * is a merge thread and if its merge operation has been aborted.
     *
     * <p>When the callback detects an aborted merge, it signals Faiss to interrupt
     * long-running operations like index creation or training, preventing resource
     * waste and ensuring timely cleanup.
     *
     * @see org.apache.lucene.index.MergeAbortChecker#isMergeAborted()
     */
    public static native void setMergeInterruptCallback();

    /**
     * Initializes a Faiss BBQ (Binary Quantized) HNSW index in native (C++) off-heap memory.
     *
     * <p>This is the first step of the BBQ index build pipeline. It allocates and configures:
     * <ul>
     *   <li>A Faiss HNSW graph structure (adjacency lists, entry point, max level) parameterized
     *       by M and efConstruction from {@code indexParameters}</li>
     *   <li>Off-heap storage for quantized vectors and their correction factors, sized to hold
     *       {@code totalLiveDocs} vectors of {@code quantizedVecBytes} bytes each plus 16 bytes
     *       of correction factors per vector (lowerInterval, upperInterval, additionalCorrection,
     *       quantizedComponentSum)</li>
     *   <li>The centroid dot-product ({@code centroidDp}), a precomputed scalar that is baked into
     *       the index and used as a correction term in the ADC scoring formula:
     *       {@code score += queryCorrection + indexCorrection - centroidDp}</li>
     * </ul>
     *
     * <p>After this call, the index is empty — no vectors or doc IDs have been added yet.
     * The caller must subsequently invoke {@link #passBBQVectorsWithCorrectionFactors} to transfer
     * quantized vectors, then {@link #addDocsToBBQIndex} to build the HNSW graph.
     *
     * <p>The returned memory address is a raw pointer to the native index object and must be
     * passed to all subsequent JNI calls. The caller is responsible for ensuring the index is
     * eventually freed (either via {@link #writeIndex} which frees after serialization, or via
     * explicit deallocation on error).
     *
     * @param totalLiveDocs    total number of vectors to be indexed; used to pre-allocate storage
     * @param dimension        dimensionality of the original float vectors (before quantization).
     *                         The quantized binary code size is derived as ((dimension + 63) / 64) * 64 / 8 bytes
     * @param indexParameters  engine-specific HNSW parameters (e.g., "m", "ef_construction", "ef_search",
     *                         "space_type"). These are passed through to Faiss's HNSW constructor
     * @param centroidDp       precomputed centroid dot-product, a scalar correction term used in the
     *                         asymmetric distance computation (ADC) during both graph construction and search
     * @param quantizedVecBytes byte length of a single 1-bit quantized vector, always 64-bit aligned
     * @return memory address (pointer) of the newly allocated native Faiss BBQ index
     */
    public static native long initFaissBBQIndex(
        final int totalLiveDocs,
        final int dimension,
        final Map<String, Object> indexParameters,
        final float centroidDp,
        final int quantizedVecBytes
    );

    /**
     * Adds a batch of document IDs to the Faiss BBQ index and triggers HNSW graph insertion.
     *
     * <p>This method is called during Phase 2 of the BBQ build pipeline, after all quantized
     * vectors and correction factors have been transferred via
     * {@link #passBBQVectorsWithCorrectionFactors}. It serves two purposes:
     * <ol>
     *   <li>Populates the FaissIdMapIndex layer that maps vector ordinals to document IDs.
     *       This mapping is critical for sparse and nested document cases where
     *       doc ID != vector ordinal (e.g., nested fields where multiple vectors share
     *       a parent document)</li>
     *   <li>Triggers HNSW insertion for each vector in the batch. The native code references
     *       quantized vectors by ordinal ({@code numAdded + offset}) from the off-heap storage
     *       that was populated in Phase 1, computing distances via the ADC formula to find
     *       neighbors during graph construction</li>
     * </ol>
     *
     * <p>This method is called repeatedly with batches of doc IDs (typically 1024 per batch)
     * until all documents have been added. The batching balances JNI call overhead against
     * memory usage for the temporary int[] array on the Java side.
     *
     * @param indexMemoryAddress pointer to the native Faiss BBQ index (returned by {@link #initFaissBBQIndex})
     * @param docIds             array of document IDs for this batch; only the first {@code numDocs}
     *                           entries are read
     * @param numDocs            number of valid document IDs in the {@code docIds} array (may be less
     *                           than docIds.length for the final batch)
     * @param numAdded           cumulative count of vectors already added before this batch; used as the
     *                           starting ordinal offset so the native code can locate the correct
     *                           quantized vectors in off-heap storage
     */
    public static native void addDocsToBBQIndex(long indexMemoryAddress, int[] docIds, int numDocs, int numAdded);

    /**
     * Transfers a batch of binary quantized vectors and their correction factors from Java heap
     * to the native Faiss BBQ index's off-heap memory.
     *
     * <p>This method is called during Phase 1 of the BBQ build pipeline, before any HNSW graph
     * construction begins. It must be called repeatedly until all vectors have been transferred.
     * Vectors are stored sequentially by ordinal in the native index, so the order of transfer
     * determines the ordinal assignment.
     *
     * <h3>Buffer layout</h3>
     * The {@code buffer} contains {@code numElements} tightly packed vector blocks, each with the
     * following little-endian layout:
     * <pre>
     * [binaryCode (quantizedVecBytes)] [lowerInterval (4B float)] [upperInterval (4B float)]
     * [additionalCorrection (4B float)] [quantizedComponentSum (4B int)]
     * </pre>
     *
     * <h3>Correction factors</h3>
     * Each vector's correction factors are used in the ADC scoring formula during HNSW
     * graph construction and search:
     * <ul>
     *   <li>{@code lowerInterval} (ax): lower bound of the scalar quantization interval</li>
     *   <li>{@code upperInterval}: upper bound; effective scale = upperInterval - lowerInterval</li>
     *   <li>{@code additionalCorrection}: residual correction for quantization error</li>
     *   <li>{@code quantizedComponentSum} (x1): sum of quantized component values, used in the
     *       cross-term {@code ay * lx * x1} of the scoring formula</li>
     * </ul>
     *
     * <p>Note: The Java side serializes quantizedComponentSum as a 4-byte int for simplicity.
     * The native SIMD search layer stores it as a 2-byte uint16_t to optimize memory layout
     * for cache-friendly vectorized access. This conversion is handled transparently by the
     * native code during the transfer.
     *
     * @param indexMemoryAddress pointer to the native Faiss BBQ index (returned by {@link #initFaissBBQIndex})
     * @param buffer             byte array containing packed [binaryCode + correctionFactors] blocks
     *                           in little-endian order; sized for the maximum batch but only the first
     *                           {@code numElements} blocks are consumed
     * @param numElements number of vector blocks (quantized vector + correction factors) in this batch to transfer
     */
    public static native void passBBQVectorsWithCorrectionFactors(long indexMemoryAddress, byte[] buffer, int numElements);
}
