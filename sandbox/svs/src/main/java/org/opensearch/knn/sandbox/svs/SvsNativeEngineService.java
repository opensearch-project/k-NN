/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.index.engine.NativeIndexBuildParams;
import org.opensearch.knn.index.engine.NativeSearchParams;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import static org.opensearch.knn.sandbox.svs.SVSConstants.INDEX_THREAD_QTY_KEY;

/**
 * {@link NativeEngineService} for the Intel SVS engine: routes every native index operation to
 * {@link SvsService} (the isolated {@code libopensearchknn_svs}).
 */
public class SvsNativeEngineService extends AbstractNativeEngineService {

    public SvsNativeEngineService() {
        super(SVSConstants.SVS_ENGINE_NAME);
    }

    @Override
    public long initIndex(NativeIndexBuildParams params) {
        return SvsService.initIndex(params.numDocs(), params.dim(), params.engineParameters().raw());
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, long indexAddress, NativeIndexBuildParams params) {
        int threadCount = params.engineParameters().get(INDEX_THREAD_QTY_KEY, 0);
        SvsService.insertToIndex(docs, vectorsAddress, params.dim(), indexAddress, threadCount);
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, NativeIndexBuildParams params) {
        SvsService.writeIndex(indexAddress, output);
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, NativeIndexBuildParams params) {
        return SvsService.loadIndexWithStream(readStream);
    }

    @Override
    public KNNQueryResult[] queryIndex(long indexPointer, NativeSearchParams params) {
        rejectParentIds(params);
        if (ArrayUtils.isNotEmpty(params.filteredIds())) {
            return SvsService.queryIndexWithFilter(
                indexPointer,
                params.queryVector(),
                params.k(),
                params.methodParameters().raw(),
                params.filteredIds(),
                params.filterIdsType()
            );
        }
        return SvsService.queryIndex(indexPointer, params.queryVector(), params.k(), params.methodParameters().raw());
    }

    @Override
    public KNNQueryResult[] radiusQueryIndex(long indexPointer, NativeSearchParams params) {
        rejectParentIds(params);
        final float radius = params.radius();
        if (radius <= 0) {
            // Backstop: SvsLibrary rejects this at query build; SVS requires a strictly positive radius.
            throw new UnsupportedOperationException(
                "The SVS engine does not support radial thresholds that resolve to a non-positive radius "
                    + "(converted radius was "
                    + radius
                    + "); use a stricter max_distance/min_score for this space type"
            );
        }
        if (ArrayUtils.isNotEmpty(params.filteredIds())) {
            return SvsService.radiusQueryIndexWithFilter(
                indexPointer,
                params.queryVector(),
                radius,
                params.methodParameters().raw(),
                params.indexMaxResultWindow(),
                params.filteredIds(),
                params.filterIdsType()
            );
        }
        return SvsService.radiusQueryIndex(
            indexPointer,
            params.queryVector(),
            radius,
            params.methodParameters().raw(),
            params.indexMaxResultWindow()
        );
    }

    @Override
    public void free(long indexPointer) {
        SvsService.free(indexPointer);
    }

    private static void rejectParentIds(NativeSearchParams params) {
        if (ArrayUtils.isNotEmpty(params.parentIds())) {
            throw new UnsupportedOperationException("Nested fields are not supported by the experimental SVS engine");
        }
    }
}
