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

import org.apache.commons.lang.ArrayUtils;
import org.opensearch.common.Nullable;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.util.IndexUtil;

import java.util.Locale;
import java.util.Map;

/**
 * Service to distribute requests to the proper engine jni service
 */
public class JNIService {
    /**
     * Initialize an index for the native library. Takes in numDocs to
     * allocate the correct amount of memory.
     *
     * @param numDocs    number of documents to be added
     * @param dim        dimension of the vector to be indexed
     * @param parameters parameters to build index
     * @param knnEngine  knn engine
     * @return address of the index in memory
     */
    public static long initIndex(long numDocs, int dim, Map<String, Object> parameters, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, parameters)) {
                return FaissService.initBinaryIndex(numDocs, dim, parameters);
            }
            if (IndexUtil.isByteIndex(parameters)) {
                return FaissService.initByteIndex(numDocs, dim, parameters);
            }

            return FaissService.initIndex(numDocs, dim, parameters);

        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "initIndexFromScratch not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Inserts to a faiss index.
     *
     * @param docs           ids of documents
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dimension      dimension of the vector to be indexed
     * @param parameters     parameters to build index
     * @param indexAddress   address of native memory where index is stored
     * @param knnEngine      knn engine
     */
    public static void insertToIndex(
        int[] docs,
        long vectorsAddress,
        int dimension,
        Map<String, Object> parameters,
        long indexAddress,
        KNNEngine knnEngine
    ) {
        int threadCount = (int) parameters.getOrDefault(KNNConstants.INDEX_THREAD_QTY, 0);
        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, parameters)) {
                FaissService.insertToBinaryIndex(docs, vectorsAddress, dimension, indexAddress, threadCount);
            } else if (IndexUtil.isByteIndex(parameters)) {
                FaissService.insertToByteIndex(docs, vectorsAddress, dimension, indexAddress, threadCount);
            } else {
                FaissService.insertToIndex(docs, vectorsAddress, dimension, indexAddress, threadCount);
            }
            return;
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "insertToIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Writes a faiss index to disk.
     *
     * @param indexPath    path to save index to
     * @param indexAddress address of native memory where index is stored
     * @param knnEngine    knn engine
     * @param parameters   parameters to build index
     */
    public static void writeIndex(String indexPath, long indexAddress, KNNEngine knnEngine, Map<String, Object> parameters) {
        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, parameters)) {
                FaissService.writeBinaryIndex(indexAddress, indexPath);
            } else if (IndexUtil.isByteIndex(parameters)) {
                FaissService.writeByteIndex(indexAddress, indexPath);
            } else {
                FaissService.writeIndex(indexAddress, indexPath);
            }
            return;
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "writeIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Create an index for the native library. The memory occupied by the vectorsAddress will be freed up during the
     * function call. So Java layer doesn't need to free up the memory. This is not an ideal behavior because Java layer
     * created the memory address and that should only free up the memory. We are tracking the proper fix for this on this
     * <a href="https://github.com/opensearch-project/k-NN/issues/1600">issue</a>
     *
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param indexPath      path to save index file to
     * @param parameters     parameters to build index
     * @param knnEngine      engine to build index for
     */
    public static void createIndex(
        int[] ids,
        long vectorsAddress,
        int dim,
        String indexPath,
        Map<String, Object> parameters,
        KNNEngine knnEngine
    ) {

        if (KNNEngine.NMSLIB == knnEngine) {
            NmslibService.createIndex(ids, vectorsAddress, dim, indexPath, parameters);
            return;
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "CreateIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Create an index for the native library with a provided template index
     *
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of vectors to be indexed
     * @param indexPath      path to save index file to
     * @param templateIndex  empty template index
     * @param parameters     parameters to build index
     * @param knnEngine      engine to build index for
     */
    public static void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        String indexPath,
        byte[] templateIndex,
        Map<String, Object> parameters,
        KNNEngine knnEngine
    ) {
        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, parameters)) {
                FaissService.createBinaryIndexFromTemplate(ids, vectorsAddress, dim, indexPath, templateIndex, parameters);
                return;
            }
            if (IndexUtil.isByteIndex(parameters)) {
                FaissService.createByteIndexFromTemplate(ids, vectorsAddress, dim, indexPath, templateIndex, parameters);
                return;
            }

            FaissService.createIndexFromTemplate(ids, vectorsAddress, dim, indexPath, templateIndex, parameters);
            return;

        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "CreateIndexFromTemplate not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Load an index into memory
     *
     * @param indexPath  path to index file
     * @param parameters parameters to be used when loading index
     * @param knnEngine  engine to load index
     * @return pointer to location in memory the index resides in
     */
    public static long loadIndex(String indexPath, Map<String, Object> parameters, KNNEngine knnEngine) {
        if (KNNEngine.NMSLIB == knnEngine) {
            return NmslibService.loadIndex(indexPath, parameters);
        }

        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, parameters)) {
                return FaissService.loadBinaryIndex(indexPath);
            } else {
                return FaissService.loadIndex(indexPath);
            }
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "LoadIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Load an index via Lucene's IndexInput.
     *
     * @param readStream A wrapper having Lucene's IndexInput to load bytes from a file.
     * @param parameters Parameters to be used when loading index
     * @param knnEngine  Engine to load index
     * @return Pointer to location in memory the index resides in
     */
    public static long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, parameters)) {
                return FaissService.loadBinaryIndexWithStream(readStream);
            } else {
                return FaissService.loadIndexWithStream(readStream);
            }
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "LoadIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Determine if index contains shared state. Currently, we cannot do this in the plugin because we do not store the
     * model definition anywhere. Only faiss supports indices that have shared state. So for all other engines it will
     * return false.
     *
     * @param indexAddr address of index to be checked.
     * @param knnEngine engine
     * @return true if index requires shared index state; false otherwise
     */
    public static boolean isSharedIndexStateRequired(long indexAddr, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            return FaissService.isSharedIndexStateRequired(indexAddr);
        }

        return false;
    }

    /**
     * Initialize the shared state for an index
     *
     * @param indexAddr address of the index to initialize from
     * @param knnEngine engine
     * @return Address of shared index state address
     */
    public static long initSharedIndexState(long indexAddr, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            return FaissService.initSharedIndexState(indexAddr);
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "InitSharedIndexState not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Set the index state for an index
     *
     * @param indexAddr           address of index to set state for
     * @param shareIndexStateAddr address of shared state to be set
     * @param knnEngine           engine
     */
    public static void setSharedIndexState(long indexAddr, long shareIndexStateAddr, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            FaissService.setSharedIndexState(indexAddr, shareIndexStateAddr);
            return;
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "SetSharedIndexState not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Query an index
     *
     * @param indexPointer     pointer to index in memory
     * @param queryVector      vector to be used for query
     * @param k                neighbors to be returned
     * @param methodParameters method parameter
     * @param knnEngine        engine to query index
     * @param filteredIds      array of ints on which should be used for search.
     * @param filterIdsType    how to filter ids: Batch or BitMap
     * @return KNNQueryResult array of k neighbors
     */
    public static KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        @Nullable Map<String, ?> methodParameters,
        KNNEngine knnEngine,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        if (KNNEngine.NMSLIB == knnEngine) {
            return NmslibService.queryIndex(indexPointer, queryVector, k, methodParameters);
        }

        if (KNNEngine.FAISS == knnEngine) {
            // This code assumes that if filteredIds == null / filteredIds.length == 0 if filter is specified then empty
            // k-NN results are already returned. Otherwise, it's a filter case and we need to run search with
            // filterIds. FilterIds is coming as empty then its the case where we need to do search with Faiss engine
            // normally.
            if (ArrayUtils.isNotEmpty(filteredIds)) {
                return FaissService.queryIndexWithFilter(
                    indexPointer,
                    queryVector,
                    k,
                    methodParameters,
                    filteredIds,
                    filterIdsType,
                    parentIds
                );
            }
            return FaissService.queryIndex(indexPointer, queryVector, k, methodParameters, parentIds);
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "QueryIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Query a binary index
     *
     * @param indexPointer     pointer to index in memory
     * @param queryVector      vector to be used for query
     * @param k                neighbors to be returned
     * @param methodParameters method parameter
     * @param knnEngine        engine to query index
     * @param filteredIds      array of ints on which should be used for search.
     * @param filterIdsType    how to filter ids: Batch or BitMap
     * @return KNNQueryResult array of k neighbors
     */
    public static KNNQueryResult[] queryBinaryIndex(
        long indexPointer,
        byte[] queryVector,
        int k,
        @Nullable Map<String, ?> methodParameters,
        KNNEngine knnEngine,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        if (KNNEngine.FAISS == knnEngine) {
            return FaissService.queryBinaryIndexWithFilter(
                indexPointer,
                queryVector,
                k,
                methodParameters,
                ArrayUtils.isEmpty(filteredIds) ? null : filteredIds,
                filterIdsType,
                parentIds
            );
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "QueryBinaryIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Free native memory pointer
     *
     * @param indexPointer location to be freed
     * @param knnEngine    engine to perform free
     */
    public static void free(final long indexPointer, final KNNEngine knnEngine) {
        free(indexPointer, knnEngine, false);
    }

    /**
     * Free native memory pointer
     *
     * @param indexPointer  location to be freed
     * @param knnEngine     engine to perform free
     * @param isBinaryIndex indicate if it is binary index or not
     */
    public static void free(final long indexPointer, final KNNEngine knnEngine, final boolean isBinaryIndex) {
        if (KNNEngine.NMSLIB == knnEngine) {
            NmslibService.free(indexPointer);
            return;
        }

        if (KNNEngine.FAISS == knnEngine) {
            FaissService.free(indexPointer, isBinaryIndex);
            return;
        }

        throw new IllegalArgumentException(String.format(Locale.ROOT, "Free not supported for provided engine : %s", knnEngine.getName()));
    }

    /**
     * Deallocate memory of the shared index state
     *
     * @param shareIndexStateAddr address of shared state
     * @param knnEngine           engine
     */
    public static void freeSharedIndexState(long shareIndexStateAddr, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            FaissService.freeSharedIndexState(shareIndexStateAddr);
            return;
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "FreeSharedIndexState not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * Train an empty index
     *
     * @param indexParameters     parameters used to build index
     * @param dimension           dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @param knnEngine           engine to perform the training
     * @return bytes array of trained template index
     */
    public static byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer, KNNEngine knnEngine) {
        if (KNNEngine.FAISS == knnEngine) {
            if (IndexUtil.isBinaryIndex(knnEngine, indexParameters)) {
                return FaissService.trainBinaryIndex(indexParameters, dimension, trainVectorsPointer);
            }
            if (IndexUtil.isByteIndex(indexParameters)) {
                return FaissService.trainByteIndex(indexParameters, dimension, trainVectorsPointer);
            }
            return FaissService.trainIndex(indexParameters, dimension, trainVectorsPointer);
        }

        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "TrainIndex not supported for provided engine : %s", knnEngine.getName())
        );
    }

    /**
     * <p>
     * The function is deprecated. Use {@link JNICommons#storeVectorData(long, float[][], long, boolean)}
     * </p>
     * Transfer vectors from Java to native
     *
     * @param vectorsPointer pointer to vectors in native memory. Should be 0 to create vector as well
     * @param trainingData   data to be transferred
     * @return pointer to native memory location of training data
     */
    @Deprecated(since = "2.14.0", forRemoval = true)
    public static long transferVectors(long vectorsPointer, float[][] trainingData) {
        return FaissService.transferVectors(vectorsPointer, trainingData);
    }

    /**
     * Range search index for a given query vector
     *
     * @param indexPointer         pointer to index in memory
     * @param queryVector          vector to be used for query
     * @param radius               search within radius threshold
     * @param methodParameters     parameters to be used when loading index
     * @param knnEngine            engine to query index
     * @param indexMaxResultWindow maximum number of results to return
     * @param filteredIds          list of doc ids to include in the query result
     * @param filterIdsType        how to filter ids: Batch or BitMap
     * @param parentIds            parent ids of the vectors
     * @return KNNQueryResult array of neighbors within radius
     */
    public static KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        @Nullable Map<String, ?> methodParameters,
        KNNEngine knnEngine,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        if (KNNEngine.FAISS == knnEngine) {
            if (ArrayUtils.isNotEmpty(filteredIds)) {
                return FaissService.rangeSearchIndexWithFilter(
                    indexPointer,
                    queryVector,
                    radius,
                    methodParameters,
                    indexMaxResultWindow,
                    filteredIds,
                    filterIdsType,
                    parentIds
                );
            }
            return FaissService.rangeSearchIndex(indexPointer, queryVector, radius, methodParameters, indexMaxResultWindow, parentIds);
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "RadiusQueryIndex not supported for provided engine"));
    }
}
