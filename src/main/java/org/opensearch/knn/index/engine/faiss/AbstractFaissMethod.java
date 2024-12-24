/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.*;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;

import java.util.Objects;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SIGNED_BYTE_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ8;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_INT8;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.index.engine.faiss.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.isFaissSQClipToFP16RangeEnabled;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.isFaissSQfp16;

public abstract class AbstractFaissMethod extends AbstractKNNMethod {

    /**
     * Constructor for the AbstractFaissMethod class.
     *
     * @param methodComponent        The method component used to create the method
     * @param spaces                 The set of spaces supported by the method
     * @param knnLibrarySearchContext The KNN library search context
     */
    public AbstractFaissMethod(MethodComponent methodComponent, Set<SpaceType> spaces, KNNLibrarySearchContext knnLibrarySearchContext) {
        super(methodComponent, spaces, knnLibrarySearchContext);
    }

    @Override
    protected PerDimensionValidator doGetPerDimensionValidator(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();
        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }

        if (VectorDataType.FLOAT == vectorDataType) {
            if (isFaissSQfp16(knnMethodContext.getMethodComponentContext())) {
                return FaissFP16Util.FP16_VALIDATOR;
            }
            return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
        }

        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    @Override
    protected PerDimensionProcessor doGetPerDimensionProcessor(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();

        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        if (VectorDataType.FLOAT == vectorDataType) {
            if (isFaissSQClipToFP16RangeEnabled(knnMethodContext.getMethodComponentContext())) {
                return FaissFP16Util.CLIP_TO_FP16_PROCESSOR;
            }
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    static KNNLibraryIndexingContext adjustIndexDescription(
        MethodAsMapBuilder methodAsMapBuilder,
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        String prefix = "";
        MethodComponentContext encoderContext = getEncoderMethodComponent(methodComponentContext);
        // We need to update the prefix used to create the faiss index if we are using the quantization
        // framework
        if (encoderContext != null && Objects.equals(encoderContext.getName(), QFrameBitEncoder.NAME)) {
            prefix = FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;
        }

        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BINARY) {
            prefix = FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;
        }

        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BYTE) {

            // If VectorDataType is Byte using Faiss engine then manipulate Index Description to use "SQ8_direct_signed" scalar quantizer
            // For example, Index Description "HNSW16,Flat" will be updated as "HNSW16,SQ8_direct_signed"
            methodAsMapBuilder.indexDescription = updateIndexDescription(methodAsMapBuilder.indexDescription, FAISS_SIGNED_BYTE_SQ);
        }

        // If SQ encoder type is 'int8' or the compression level is 4x then manipulate Index Description to use "SQ8" scalar quantizer
        // For example, Index Description "HNSW16,Flat" will be updated as "HNSW16,SQ8"
        if (encoderContext != null
            && Objects.equals(encoderContext.getName(), ENCODER_SQ)
            && Objects.equals(encoderContext.getParameters().getOrDefault(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16), FAISS_SQ_ENCODER_INT8)) {
            methodAsMapBuilder.indexDescription = updateIndexDescription(methodAsMapBuilder.indexDescription, FAISS_SQ8);
        }
        methodAsMapBuilder.indexDescription = prefix + methodAsMapBuilder.indexDescription;
        return methodAsMapBuilder.build();
    }

    private static String updateIndexDescription(String indexDescription, String indexDescriptionName) {
        if (StringUtils.isEmpty(indexDescription)) {
            return indexDescription;
        }

        StringBuilder indexDescriptionBuilder = new StringBuilder();
        indexDescriptionBuilder.append(indexDescription.split(",")[0]);
        indexDescriptionBuilder.append(",");
        indexDescriptionBuilder.append(indexDescriptionName);
        return indexDescriptionBuilder.toString();
    }

    static MethodComponentContext getEncoderMethodComponent(MethodComponentContext methodComponentContext) {
        if (!methodComponentContext.getParameters().containsKey(METHOD_ENCODER_PARAMETER)) {
            return null;
        }
        Object object = methodComponentContext.getParameters().get(METHOD_ENCODER_PARAMETER);
        if (!(object instanceof MethodComponentContext)) {
            return null;
        }
        return (MethodComponentContext) object;
    }

    @Override
    protected SpaceType convertUserToMethodSpaceType(SpaceType spaceType) {
        // While FAISS doesn't directly support cosine similarity, we can leverage the mathematical
        // relationship between cosine similarity and inner product for normalized vectors to add support.
        // When ||a|| = ||b|| = 1, cos(x) = a dot b
        if (spaceType == SpaceType.COSINESIMIL) {
            return SpaceType.INNER_PRODUCT;
        }
        return super.convertUserToMethodSpaceType(spaceType);
    }

    @Override
    protected VectorTransformer getVectorTransformer(SpaceType spaceType) {
        return VectorTransformerFactory.getVectorTransformer(KNNEngine.FAISS, spaceType);
    }
}
