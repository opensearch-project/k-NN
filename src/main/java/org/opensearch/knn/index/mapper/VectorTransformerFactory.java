/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;

/**
 * Factory class responsible for creating appropriate vector transformers based on the KNN method context.
 * This factory determines whether vectors need transformation based on the engine type and space type.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class VectorTransformerFactory {

    /**
     * Returns a vector transformer based on the provided KNN method context.
     * For FAISS engine with cosine similarity space type, returns a NormalizeVectorTransformer
     * since FAISS doesn't natively support cosine space type. For all other cases,
     * returns a no-operation transformer.
     *
     * @param context The KNN method context containing engine and space type information
     * @return VectorTransformer An appropriate vector transformer instance
     * @throws IllegalArgumentException if the context parameter is null
     */
    public static VectorTransformer getVectorTransformer(final KNNMethodContext context) {
        if (context == null) {
            throw new IllegalArgumentException("KNNMethod context cannot be null");
        }
        return getVectorTransformer(context.getKnnEngine(), context.getSpaceType());
    }

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
        return shouldNormalizeVector(knnEngine, spaceType) ? new NormalizeVectorTransformer() : VectorTransformer.NOOP_VECTOR_TRANSFORMER;
    }

    private static boolean shouldNormalizeVector(final KNNEngine knnEngine, final SpaceType spaceType) {
        return knnEngine == KNNEngine.FAISS && spaceType == SpaceType.COSINESIMIL;
    }
}
