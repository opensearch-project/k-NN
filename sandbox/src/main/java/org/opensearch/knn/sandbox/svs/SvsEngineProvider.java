/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.SandboxEngineProvider;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.util.Map;

/**
 * {@link SandboxEngineProvider} for the experimental Intel SVS engine. Discovered by {@code KNNEngine} via
 * {@code META-INF/services}, it gives the generic {@code KNNEngine.EXPERIMENTAL} slot its identity:
 * the engine name {@code "svs"}, the {@link SvsLibrary} (methods/extension/scoring), and the native index
 * lifecycle — all delegated to {@link SvsService} (the isolated {@code libopensearchknn_svs}).
 *
 * <p>{@link #engineName()} and {@link #library()} are evaluated when {@code KNNEngine} initializes; they do
 * not touch {@link SvsService}, so discovery alone never loads the native library (which loads lazily on the
 * first native call).
 *
 * <p>Operations outside the SVS scope (template builds, radial, nested queries) throw
 * {@link UnsupportedOperationException}; the corresponding capability sets in {@code KNNEngine} already
 * exclude this engine, so these are defensive backstops.
 */
public class SvsEngineProvider implements SandboxEngineProvider {

    @Override
    public String engineName() {
        return SVSConstants.SVS_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return SvsLibrary.INSTANCE;
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        return SvsService.initIndex(numDocs, dim, parameters);
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        int threadCount = (int) parameters.getOrDefault(KNNConstants.INDEX_THREAD_QTY, 0);
        SvsService.insertToIndex(docs, vectorsAddress, dimension, indexAddress, threadCount);
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        SvsService.writeIndex(indexAddress, output);
    }

    @Override
    public void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    ) {
        throw new UnsupportedOperationException("Template-based index builds are not supported by the experimental SVS engine");
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        return SvsService.loadIndexWithStream(readStream);
    }

    @Override
    public KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        if (ArrayUtils.isNotEmpty(parentIds)) {
            // Reject rather than silently ignore parentIds, which would return multiple children per parent.
            throw new UnsupportedOperationException("Nested fields are not supported by the experimental SVS engine");
        }
        if (ArrayUtils.isNotEmpty(filteredIds)) {
            return SvsService.queryIndexWithFilter(indexPointer, queryVector, k, methodParameters, filteredIds, filterIdsType);
        }
        return SvsService.queryIndex(indexPointer, queryVector, k, methodParameters);
    }

    @Override
    public KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        throw new UnsupportedOperationException("Radial search is not supported by the experimental SVS engine");
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        // isBinaryIndex is ignored: the SVS engine has no binary indices.
        SvsService.free(indexPointer);
    }
}
