/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;

/**
 * Interface for resolving the appropriate {@link KnnVectorsFormat} for a given field at codec format construction time.
 * Each engine provides its own implementation to encapsulate format construction logic.
 */
public interface CodecFormatResolver {

    /**
     * Resolves the appropriate {@link KnnVectorsFormat} for a given field.
     *
     * @param field                 the field name
     * @param methodContext         the KNN method context (engine, space type, method component); may be null for model-based fields
     * @param params                the method component parameters; may be null
     * @param defaultMaxConnections default max connections for HNSW
     * @param defaultBeamWidth      default beam width for HNSW
     * @return the resolved {@link KnnVectorsFormat}
     */
    KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth
    );

    /**
     * Overload that additionally threads the resolved {@link CompressionLevel} through. Needed by
     * the Lucene FLAT format factory to pick the correct scalar-quantization encoding when the
     * user selects {@code method=flat} with x8 / x16 / x32 compression. Default implementation
     * delegates to {@link #resolve(String, KNNMethodContext, Map, int, int)} so engines that do
     * not consume the compression level (e.g., Faiss) don't need to override.
     */
    default KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth,
        CompressionLevel compressionLevel
    ) {
        return resolve(field, methodContext, params, defaultMaxConnections, defaultBeamWidth);
    }

    KnnVectorsFormat resolve();
}
