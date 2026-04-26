/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.util.HashMap;
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

    /**
     * Returns the {@link VectorTransformer} for a field by reading the required metadata directly
     * from the {@link FieldInfo}. Intended for codec-layer callers that do not have access to a
     * resolved {@link MethodComponentContext}.
     *
     * <p>Space type resolution:
     * <ul>
     *   <li>Primary source: {@link FieldInfo#getAttribute(String)} with key {@link KNNConstants#SPACE_TYPE}.</li>
     *   <li>Fallback: {@link ModelMetadata#getSpaceType()} via {@link ModelUtil#getModelMetadata(String)} when
     *       the field is model-based. Reads cluster state (no network I/O).</li>
     * </ul>
     *
     * <p>{@link MethodComponentContext} resolution:
     * <ul>
     *   <li>Primary source: {@link FieldInfo#getAttribute(String)} with key {@link KNNConstants#PARAMETERS}.
     *       The attribute value is the JSON serialization of
     *       {@code KNNLibraryIndexingContext.getLibraryParameters()} and is parsed back into a
     *       {@link MethodComponentContext} so that Lucene-specific normalization conditions
     *       (flat / SQ 1-bit) are evaluated identically to the query-time path.</li>
     *   <li>Fallback: {@link ModelMetadata#getMethodComponentContext()} for model-based fields.</li>
     *   <li>If neither is available (e.g. legacy fields without the PARAMETERS attribute), {@code null}
     *       is used and Lucene-specific conditions are skipped.</li>
     * </ul>
     *
     * @param fieldInfo field metadata from the Lucene segment
     * @return a {@link VectorTransformer}, possibly {@link #NOOP_VECTOR_TRANSFORMER}
     */
    public static VectorTransformer getVectorTransformer(final FieldInfo fieldInfo) {
        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        final SpaceType spaceType = resolveSpaceTypeFromFieldInfo(fieldInfo);
        if (spaceType == null) {
            return NOOP_VECTOR_TRANSFORMER;
        }
        final MethodComponentContext methodCtx = resolveMethodComponentContextFromFieldInfo(fieldInfo);
        return getVectorTransformer(engine, spaceType, methodCtx);
    }

    private static SpaceType resolveSpaceTypeFromFieldInfo(final FieldInfo fieldInfo) {
        final String spaceTypeStr = fieldInfo.getAttribute(KNNConstants.SPACE_TYPE);
        if (StringUtils.isNotEmpty(spaceTypeStr)) {
            return SpaceType.getSpace(spaceTypeStr);
        }
        final String modelId = fieldInfo.getAttribute(KNNConstants.MODEL_ID);
        if (StringUtils.isNotEmpty(modelId)) {
            final ModelMetadata metadata = ModelUtil.getModelMetadata(modelId);
            return metadata != null ? metadata.getSpaceType() : null;
        }
        return null;
    }

    private static MethodComponentContext resolveMethodComponentContextFromFieldInfo(final FieldInfo fieldInfo) {
        final String parametersString = fieldInfo.getAttribute(KNNConstants.PARAMETERS);
        if (StringUtils.isNotEmpty(parametersString)) {
            try {
                final Map<String, Object> parsed = XContentHelper.createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    new BytesArray(parametersString),
                    MediaTypeRegistry.getDefaultMediaType()
                ).map();
                // The JSON written by EngineFieldMapper contains top-level keys beyond NAME/PARAMETERS
                // (e.g. space_type, vector_data_type, index_description). MethodComponentContext.parse
                // rejects unknown keys, so we narrow down to the two fields it understands before parsing.
                final Map<String, Object> methodMap = new HashMap<>();
                if (parsed.containsKey(KNNConstants.NAME)) {
                    methodMap.put(KNNConstants.NAME, parsed.get(KNNConstants.NAME));
                }
                if (parsed.containsKey(KNNConstants.PARAMETERS)) {
                    methodMap.put(KNNConstants.PARAMETERS, parsed.get(KNNConstants.PARAMETERS));
                }
                if (!methodMap.isEmpty()) {
                    return MethodComponentContext.parse(methodMap);
                }
            } catch (Exception e) {
                // If parsing fails for any reason, fall through to other resolution paths.
            }
        }
        final String modelId = fieldInfo.getAttribute(KNNConstants.MODEL_ID);
        if (StringUtils.isNotEmpty(modelId)) {
            final ModelMetadata metadata = ModelUtil.getModelMetadata(modelId);
            if (metadata != null && metadata.getMethodComponentContext() != null) {
                return metadata.getMethodComponentContext();
            }
        }
        return null;
    }

    private static boolean shouldNormalizeVector(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final MethodComponentContext methodComponentContext
    ) {
        if (spaceType != SpaceType.COSINESIMIL) {
            return false;
        }
        if (knnEngine == KNNEngine.FAISS) {
            return true;
        }
        if (knnEngine == KNNEngine.LUCENE) {
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
