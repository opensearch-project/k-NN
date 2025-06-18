/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Factory class responsible for creating appropriate vector transformers.
 * This factory determines whether vectors need transformation based on the engine type and space type.
 */
@Log4j2
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
    public static VectorTransformer getVectorTransformer(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        boolean isRandomRotation,
        int dimension
    ) {
        log.info("in factory method");
        boolean shouldNormalizeVector = shouldNormalizeVector(knnEngine, spaceType);
        if (shouldNormalizeVector && isRandomRotation) return new RandomRotationNormalizeVectorTransformer(dimension);
        if (shouldNormalizeVector) return new NormalizeVectorTransformer();
        if (isRandomRotation) return new RandomRotationVectorTransformer(dimension);
        return NOOP_VECTOR_TRANSFORMER;
    }

    private static boolean shouldNormalizeVector(final KNNEngine knnEngine, final SpaceType spaceType) {
        return knnEngine == KNNEngine.FAISS && spaceType == SpaceType.COSINESIMIL;
    }
}
