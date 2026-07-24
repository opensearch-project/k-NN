/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.Value;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;

/**
 * Immutable context object passed to Lucene format factories registered in
 * {@link KNN1040BasePerFieldKnnVectorsFormat}. Contains everything a factory needs
 * to construct the appropriate
 * {@link org.apache.lucene.codecs.KnnVectorsFormat}.
 */
@Value
public class KnnVectorsFormatContext {
    /**
     * The field name being processed (useful for logging).
     */
    String field;

    /**
     * The KNN method context containing engine, space type, and method component
     * info.
     */
    KNNMethodContext methodContext;

    /**
     * The method component parameters (may be null).
     */
    Map<String, Object> params;

    /**
     * Default max connections if not specified in params.
     */
    int defaultMaxConnections;

    /**
     * Default beam width if not specified in params.
     */
    int defaultBeamWidth;

    /**
     * The value of {@code index.knn.advanced.approximate_threshold} for the index. Format factories
     * should convert this to a {@code tinySegmentsThreshold} to control whether the HNSW graph is
     * built for a given segment:
     * <ul>
     *   <li>{@code 0} (default): always build the graph.</li>
     *   <li>{@code N > 0}: skip the graph when {@code docCount < N}.</li>
     *   <li>{@code -1}: never build the graph.</li>
     * </ul>
     */
    int approximateThreshold;
}
