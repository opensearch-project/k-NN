/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Factory class responsible for creating appropriate vector transformers.
 * This factory determines whether vectors need transformation based on the engine type and space type.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class VectorTransformerFactory {

    /**
     * A no-operation transformer that returns vector values unchanged.
     */
    public final static VectorTransformer NOOP_VECTOR_TRANSFORMER = new VectorTransformer() {
    };

    /**
     * Returns a vector transformer based on the provided KNN engine and space type.
     * For FAISS engine with cosine similarity space type, returns a NormalizeVectorTransformer
     * since FAISS doesn't natively support cosine space type. For all other cases,
     * returns a no-operation transformer.
     *
     * @param knnEngine The KNN engine type
     * @param spaceType The space type
     * @return VectorTransformer An appropriate vector transformer instance
     */
    public static VectorTransformer getVectorTransformer(final KNNEngine knnEngine, final SpaceType spaceType) {
        return shouldNormalizeVector(knnEngine, spaceType) ? new NormalizeVectorTransformer() : NOOP_VECTOR_TRANSFORMER;
    }

    private static boolean shouldNormalizeVector(final KNNEngine knnEngine, final SpaceType spaceType) {
        return knnEngine == KNNEngine.FAISS && spaceType == SpaceType.COSINESIMIL;
    }
}
