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
import org.opensearch.knn.indices.ModelMetadata;

/**
 * Factory class responsible for creating appropriate vector transformers based on the KNN method context.
 * This factory determines whether vectors need transformation based on the engine type and space type.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class VectorTransformerFactory {

    /**
     * A no-operation transformer that returns vector values unchanged.
     */
    private final static VectorTransformer NOOP_VECTOR_TRANSFORMER = new VectorTransformer() {
    };

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
     * Returns a vector transformer instance for vector transformations.
     * This method provides access to the default no-operation vector transformer
     * that performs identity transformation on vectors. The transformer does not
     * modify the input vectors and returns them as-is.This implementation returns a stateless, thread-safe transformer
     * instance that can be safely shared across multiple calls
     *
     * @return VectorTransformer A singleton instance of the no-operation vector
     *         transformer (NOOP_VECTOR_TRANSFORMER)
     */
    public static VectorTransformer getVectorTransformer() {
        return NOOP_VECTOR_TRANSFORMER;
    }

    /**
     * Creates a VectorTransformer based on the provided model metadata.
     *
     * @param metadata The model metadata containing KNN engine and space type configuration.
     *                This parameter must not be null.
     * @return A VectorTransformer instance configured according to the model metadata
     * @throws IllegalArgumentException if metadata is null
     *
     * The factory determines the appropriate transformer implementation based on:
     * - The KNN engine (e.g., FAISS, NMSLIB)
     * - The space type (e.g., L2, COSINE)
     *
     * The returned transformer can be used to modify vectors in-place according to
     * the specified engine and space type requirements.
     */
    public static VectorTransformer getVectorTransformer(final ModelMetadata metadata) {
        if (metadata == null) {
            throw new IllegalArgumentException("ModelMetadata cannot be null");
        }
        return getVectorTransformer(metadata.getKnnEngine(), metadata.getSpaceType());
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
    private static VectorTransformer getVectorTransformer(final KNNEngine knnEngine, final SpaceType spaceType) {
        return shouldNormalizeVector(knnEngine, spaceType) ? new NormalizeVectorTransformer() : getVectorTransformer();
    }

    private static boolean shouldNormalizeVector(final KNNEngine knnEngine, final SpaceType spaceType) {
        return knnEngine == KNNEngine.FAISS && spaceType == SpaceType.COSINESIMIL;
    }
}
