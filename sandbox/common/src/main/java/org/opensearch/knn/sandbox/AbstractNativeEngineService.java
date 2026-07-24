/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.engine.NativeEngineService;

import java.util.Locale;
import java.util.Map;

/**
 * Convenience base for a sandbox tenant's {@link NativeEngineService}: every operation throws a
 * descriptive {@link UnsupportedOperationException}, so a tenant overrides only the operations its engine
 * actually supports and inherits honest rejections for the rest.
 *
 * <p>Tenants are not required to extend this class; implementing {@link NativeEngineService} directly is
 * equally valid.
 */
public abstract class AbstractNativeEngineService implements NativeEngineService {

    /** Engine name used in the exception messages, e.g. {@code "my_engine"}. */
    private final String engineName;

    protected AbstractNativeEngineService(String engineName) {
        this.engineName = engineName;
    }

    private UnsupportedOperationException unsupported(String operation) {
        return new UnsupportedOperationException(String.format(Locale.ROOT, "%s is not supported by the %s engine", operation, engineName));
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        throw unsupported("Index building");
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        throw unsupported("Index building");
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        throw unsupported("Index building");
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
        throw unsupported("Template-based index building");
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        throw unsupported("Index loading");
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
        throw unsupported("Top-k search");
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
        throw unsupported("Radial search");
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        throw unsupported("Index freeing");
    }
}
