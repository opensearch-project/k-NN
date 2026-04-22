/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

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

    private final static NormalizeVectorTransformer DEFAULT_VECTOR_TRANSFORMER = new NormalizeVectorTransformer();

    /**
     * Returns a vector transformer based on the provided KNN engine, space type, and method component context.
     * Returns a NormalizeVectorTransformer for:
     * <ul>
     *   <li>Faiss engine with cosine similarity (Faiss doesn't natively support cosine)</li>
     *   <li>Lucene engine with cosine similarity when using SQ 1-bit encoding or flat method
     *       (these paths use {@code KNN1040ScalarQuantizedVectorScorer} which requires a unit vector)</li>
     * </ul>
     *
     * @param knnEngine The KNN engine type
     * @param spaceType The space type
     * @param methodComponentContext The method component context containing method name and parameters, may be null
     * @return VectorTransformer An appropriate vector transformer instance
     */
    public static VectorTransformer getVectorTransformer(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final MethodComponentContext methodComponentContext
    ) {
        return shouldNormalizeVector(knnEngine, spaceType, methodComponentContext) ? DEFAULT_VECTOR_TRANSFORMER : NOOP_VECTOR_TRANSFORMER;
    }

    private static boolean shouldNormalizeVector(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final MethodComponentContext methodComponentContext
    ) {
        if (spaceType != SpaceType.COSINESIMIL) {
            return false;
        }
        if (knnEngine == BuiltinKNNEngine.FAISS) {
            return true;
        }
        if (knnEngine == BuiltinKNNEngine.LUCENE) {
            return shouldNormalizeForLuceneEngine(methodComponentContext);
        }
        return false;
    }

    private static boolean shouldNormalizeForLuceneEngine(final MethodComponentContext methodComponentContext) {
        if (methodComponentContext == null) {
            return false;
        }

        if (METHOD_FLAT.equals(methodComponentContext.getName())) {
            return true;
        }
        if (isLuceneSQOneBit(methodComponentContext.getParameters())) {
            return true;
        }
        return false;
    }

    private static boolean isLuceneSQOneBit(final Map<String, Object> params) {
        if (params == null) {
            return false;
        }
        Object encoderObj = params.get(METHOD_ENCODER_PARAMETER);
        if (encoderObj instanceof MethodComponentContext == false) {
            return false;
        }
        MethodComponentContext encoderCtx = (MethodComponentContext) encoderObj;
        if (ENCODER_SQ.equals(encoderCtx.getName()) == false) {
            return false;
        }
        Object bits = encoderCtx.getParameters().get(LUCENE_SQ_BITS);
        return bits instanceof Integer && (Integer) bits == 1;
    }
}
