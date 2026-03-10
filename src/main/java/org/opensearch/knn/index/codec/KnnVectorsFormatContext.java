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
 * {@link BasePerFieldKnnVectorsFormat}. Contains everything a factory needs
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
}
