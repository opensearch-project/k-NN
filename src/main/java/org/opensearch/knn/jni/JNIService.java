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
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Map;

/**
 * Service to distribute requests to the proper engine jni service
 */
public class JNIService {

    /**
     * Create an index for the native library
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param parameters parameters to build index
     * @param engineName name of engine to build index for
     */
    public static void createIndex(int[] ids, float[][] data, String indexPath, Map<String, Object> parameters, String engineName) {
        if (KNNEngine.NMSLIB.getName().equals(engineName)) {
            NmslibService.createIndex(ids, data, indexPath, parameters);
            return;
        }

        if (KNNEngine.FAISS.getName().equals(engineName)) {
            FaissService.createIndex(ids, data, indexPath, parameters);
            return;
        }

        throw new IllegalArgumentException("CreateIndex not supported for provided engine");
    }

    /**
     * Create an index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param templateIndex empty template index
     * @param parameters parameters to build index
     * @param engineName name of engine to build index for
     */
    public static void createIndexFromTemplate(
        int[] ids,
        float[][] data,
        String indexPath,
        byte[] templateIndex,
        Map<String, Object> parameters,
        String engineName
    ) {
        if (KNNEngine.FAISS.getName().equals(engineName)) {
            FaissService.createIndexFromTemplate(ids, data, indexPath, templateIndex, parameters);
            return;
        }

        throw new IllegalArgumentException("CreateIndexFromTemplate not supported for provided engine");
    }

    /**
     * Load an index into memory
     *
     * @param indexPath path to index file
     * @param parameters parameters to be used when loading index
     * @param engineName name of engine to load index
     * @return pointer to location in memory the index resides in
     */
    public static long loadIndex(String indexPath, Map<String, Object> parameters, String engineName) {
        if (KNNEngine.NMSLIB.getName().equals(engineName)) {
            return NmslibService.loadIndex(indexPath, parameters);
        }

        if (KNNEngine.FAISS.getName().equals(engineName)) {
            return FaissService.loadIndex(indexPath);
        }

        throw new IllegalArgumentException("LoadIndex not supported for provided engine");
    }

    /**
     * Determine if index is IVFPQ with L2 metric. Currently, we cannot do this in the plugin because we
     * do not store the model definition anywhere. Only faiss supports IVFPQ indices. So for all other engines it will
     * return false.
     *
     * @param indexAddr addrees of index to be checked.
     * @param engineName name of engine
     * @return true if index is of type IVFPQ-l2; false otherwise
     */
    public static boolean isIndexIVFPQL2(long indexAddr, String engineName) {
        if (KNNEngine.FAISS.getName().equals(engineName)) {
            return FaissService.isIndexIVFPQL2(indexAddr);
        }

        return false;
    }

    /**
     * Initialize the shared state for an index
     *
     * @param indexAddr address of the index to initialize from
     * @param engineName name of engine
     * @return Address of shared index state address
     */
    public static long initSharedIndexState(long indexAddr, String engineName) {
        if (KNNEngine.FAISS.getName().equals(engineName)) {
            return FaissService.initSharedIndexState(indexAddr);
        }

        throw new IllegalArgumentException("InitSharedIndexState not supported for provided engine");
    }

    /**
     * Set the index state for an index
     *
     * @param indexAddr address of index to set state for
     * @param shareIndexStateAddr address of shared state to be set
     * @param engineName name of engine
     */
    public static void setSharedIndexState(long indexAddr, long shareIndexStateAddr, String engineName) {
        if (KNNEngine.FAISS.getName().equals(engineName)) {
            FaissService.setSharedIndexState(indexAddr, shareIndexStateAddr);
            return;
        }

        throw new IllegalArgumentException("SetSharedIndexState not supported for provided engine");
    }

    /**
     * Query an index
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector  vector to be used for query
     * @param k            neighbors to be returned
     * @param engineName   name of engine to query index
     * @param filteredIds  array of ints on which should be used for search.
     * @param filterIdsType how to filter ids: Batch or BitMap
     * @return KNNQueryResult array of k neighbors
     */
    public static KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        String engineName,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        if (KNNEngine.NMSLIB.getName().equals(engineName)) {
            return NmslibService.queryIndex(indexPointer, queryVector, k);
        }

        if (KNNEngine.FAISS.getName().equals(engineName)) {
            // This code assumes that if filteredIds == null / filteredIds.length == 0 if filter is specified then empty
            // k-NN results are already returned. Otherwise, it's a filter case and we need to run search with
            // filterIds. FilterIds is coming as empty then its the case where we need to do search with Faiss engine
            // normally.
            if (ArrayUtils.isNotEmpty(filteredIds)) {
                return FaissService.queryIndexWithFilter(indexPointer, queryVector, k, filteredIds, filterIdsType, parentIds);
            }
            return FaissService.queryIndex(indexPointer, queryVector, k, parentIds);
        }
        throw new IllegalArgumentException("QueryIndex not supported for provided engine");
    }

    /**
     * Free native memory pointer
     *
     * @param indexPointer location to be freed
     * @param engineName engine to perform free
     */
    public static void free(long indexPointer, String engineName) {
        if (KNNEngine.NMSLIB.getName().equals(engineName)) {
            NmslibService.free(indexPointer);
            return;
        }

        if (KNNEngine.FAISS.getName().equals(engineName)) {
            FaissService.free(indexPointer);
            return;
        }

        throw new IllegalArgumentException("Free not supported for provided engine");
    }

    /**
     * Deallocate memory of the shared index state
     *
     * @param shareIndexStateAddr address of shared state
     * @param engineName name of engine
     */
    public static void freeSharedIndexState(long shareIndexStateAddr, String engineName) {
        if (KNNEngine.FAISS.getName().equals(engineName)) {
            FaissService.freeSharedIndexState(shareIndexStateAddr);
            return;
        }

        throw new IllegalArgumentException("FreeSharedIndexState not supported for provided engine");
    }

    /**
     * Train an empty index
     *
     * @param indexParameters parameters used to build index
     * @param dimension dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @param engineName engine to perform the training
     * @return bytes array of trained template index
     */
    public static byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer, String engineName) {
        if (KNNEngine.FAISS.getName().equals(engineName)) {
            return FaissService.trainIndex(indexParameters, dimension, trainVectorsPointer);
        }

        throw new IllegalArgumentException("TrainIndex not supported for provided engine");
    }

    /**
     * Transfer vectors from Java to native
     *
     * @param vectorsPointer pointer to vectors in native memory. Should be 0 to create vector as well
     * @param trainingData data to be transferred
     * @return pointer to native memory location of training data
     */
    public static long transferVectors(long vectorsPointer, float[][] trainingData) {
        return FaissService.transferVectors(vectorsPointer, trainingData);
    }

    /**
     * Free vectors from memory
     *
     * @param vectorsPointer to be freed
     */
    public static void freeVectors(long vectorsPointer) {
        FaissService.freeVectors(vectorsPointer);
    }
}
