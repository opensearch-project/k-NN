/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
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
     *   <li>Lucene engine with cosine similarity when using SQ multi-bit (bits ∈ {1, 2, 4})
     *       encoding or flat method — these paths use {@code KNN1040ScalarQuantizedVectorScorer}
     *       which requires a unit vector.</li>
     *   <li>Lucene engine with cosine similarity on {@code half_float}, for any method</li>
     * </ul>
     *
     * @param knnEngine The KNN engine type
     * @param spaceType The space type
     * @param methodComponentContext The method component context containing method name and parameters, may be null
     * @param vectorDataType The vector data type, which decides normalization on its own for half_float
     * @return VectorTransformer An appropriate vector transformer instance
     */
    public static VectorTransformer getVectorTransformer(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final MethodComponentContext methodComponentContext,
        final VectorDataType vectorDataType
    ) {
        return shouldNormalizeVector(knnEngine, spaceType, methodComponentContext, vectorDataType)
            ? DEFAULT_VECTOR_TRANSFORMER
            : NOOP_VECTOR_TRANSFORMER;
    }

    private static boolean shouldNormalizeVector(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final MethodComponentContext methodComponentContext,
        final VectorDataType vectorDataType
    ) {
        if (spaceType != SpaceType.COSINESIMIL) {
            return false;
        }
        if (knnEngine == KNNEngine.FAISS) {
            return true;
        }
        if (knnEngine == KNNEngine.LUCENE) {
            return shouldNormalizeForLuceneEngine(methodComponentContext, vectorDataType);
        }
        return false;
    }

    private static boolean shouldNormalizeForLuceneEngine(
        final MethodComponentContext methodComponentContext,
        final VectorDataType vectorDataType
    ) {
        // Every half_float level stores raw fp16 through KNN1040HalfFloatFlatVectorsFormat,
        // whose scorer uses the native FP16_COSINE kernel: (1 + dot) / 2, which equals
        // cosine only for unit vectors.
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            return true;
        }
        if (methodComponentContext == null) {
            return false;
        }

        if (METHOD_FLAT.equals(methodComponentContext.getName())) {
            return true;
        }
        if (isLuceneSQMultiBit(methodComponentContext.getParameters())) {
            return true;
        }
        return false;
    }

    /**
     * True when the Lucene SQ encoder is configured with a multi-bit MOS bit width
     * (bits ∈ {1, 2, 4}). All three widths route through {@code KNN1040ScalarQuantizedVectorScorer}
     * which requires a unit-normalized query vector for cosine similarity.
     */
    private static boolean isLuceneSQMultiBit(final Map<String, Object> params) {
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
        if (bits instanceof Integer == false) {
            return false;
        }
        int b = (Integer) bits;
        return b == 1 || b == 2 || b == 4;
    }
}
