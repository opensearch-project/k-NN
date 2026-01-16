/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.core.action.ActionListener;
import org.opensearch.index.query.QueryBuilder;

public interface MMRQueryTransformer<T extends QueryBuilder> {
    /**
     * Transform the queryBuilder to oversample for MMR.
     * Also need to figure out the vector field path and the space type and set them in the MMRProcessingContext for
     * response processor to consume.
     * @param queryBuilder
     * @param listener
     * @param mmrTransformContext {@link MMRTransformContext}
     */
    void transform(T queryBuilder, ActionListener<Void> listener, MMRTransformContext mmrTransformContext);

    /**
     * @return The name of the query which will be used to find the transformer.
     */
    String getQueryName();
}
