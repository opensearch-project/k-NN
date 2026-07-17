/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.common.Nullable;

import java.util.Map;

/**
 * Interface for resolving the appropriate {@link KnnVectorsFormat} for a given field at codec format construction time.
 * Each engine provides its own implementation to encapsulate format construction logic.
 */
public interface CodecFormatResolver {

    /**
     * Resolves the appropriate {@link KnnVectorsFormat} for a given field.
     * Implementations should prefer the resolved index spec when non-null and fall back to parameter inspection otherwise.
     *
     * @param field                 the field name
     * @param methodContext         the KNN method context (engine, space type, method component); may be null for model-based fields
     * @param params                the method component parameters; may be null
     * @param defaultMaxConnections default max connections for HNSW
     * @param defaultBeamWidth      default beam width for HNSW
     * @param resolvedSpec          the resolved index spec; may be null
     * @return the resolved {@link KnnVectorsFormat}
     */
    KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth,
        @Nullable ResolvedIndexSpec resolvedSpec
    );

    KnnVectorsFormat resolve();
}
