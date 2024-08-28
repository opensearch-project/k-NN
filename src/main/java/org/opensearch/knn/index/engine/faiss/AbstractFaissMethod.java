/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.FilterKNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.config.CompressionConfig;
import org.opensearch.knn.index.engine.config.WorkloadModeConfig;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_SIGNED_BYTE_SQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.index.engine.faiss.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

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
    protected PerDimensionValidator doGetPerDimensionValidator(KNNMethodConfigContext knnMethodConfigContext) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();
        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }

        if (VectorDataType.FLOAT == vectorDataType) {
            if (isEncoderSQfp16(knnMethodConfigContext)) {
                return FaissFP16Util.FP16_VALIDATOR;
            }
            return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
        }

        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    private boolean isEncoderSQfp16(KNNMethodConfigContext knnMethodConfigContext) {
        Map<String, Object> encoderParams = getSQFP16EncoderParamsOrNull(knnMethodConfigContext);
        if (encoderParams == null) {
            return false;
        }
        return FAISS_SQ_ENCODER_FP16.equals(encoderParams.get(FAISS_SQ_TYPE));
    }

    private boolean isFaissSQClipToFP16RangeEnabled(KNNMethodConfigContext knnMethodConfigContext) {
        Map<String, Object> encoderParams = getSQFP16EncoderParamsOrNull(knnMethodConfigContext);
        if (encoderParams == null) {
            return false;
        }
        return (boolean) encoderParams.get(FAISS_SQ_CLIP);
    }

    private Map<String, Object> getSQFP16EncoderParamsOrNull(KNNMethodConfigContext knnMethodConfigContext) {
        MethodComponentContext resolveMethodComponentContext = resolveMethodComponentContext(knnMethodConfigContext);
        Map<String, Object> parameters = resolveMethodComponentContext.getParameters().orElse(null);
        if (parameters == null || parameters.containsKey(METHOD_ENCODER_PARAMETER) == false) {
            return null;
        }

        MethodComponentContext encoderContext = (MethodComponentContext) parameters.get(METHOD_ENCODER_PARAMETER);
        if (encoderContext == null || ENCODER_SQ.equals(encoderContext.getName()) == false) {
            return null;
        }

        Map<String, Object> encoderParameters = encoderContext.getParameters().orElse(null);
        if (encoderParameters == null || encoderParameters.containsKey(FAISS_SQ_TYPE) == false) {
            return null;
        }

        return encoderParameters;
    }

    @Override
    protected PerDimensionProcessor doGetPerDimensionProcessor(KNNMethodConfigContext knnMethodConfigContext) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();

        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        if (VectorDataType.FLOAT == vectorDataType) {
            if (isFaissSQClipToFP16RangeEnabled(knnMethodConfigContext)) {
                return FaissFP16Util.CLIP_TO_FP16_PROCESSOR;
            }
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    static KNNLibraryIndexingContext adjustIndexDescription(
        MethodAsMapBuilder methodAsMapBuilder,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        String prefix = "";
        // We need to update the prefix used to create the faiss index if we are using the quantization
        // framework
        if (methodAsMapBuilder.getQuantizationConfig() != null) {
            prefix = FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;
        }

        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BINARY) {
            prefix = FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;
        }
        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BYTE) {

            // If VectorDataType is Byte using Faiss engine then manipulate Index Description to use "SQ8_direct_signed" scalar quantizer
            // For example, Index Description "HNSW16,Flat" will be updated as "HNSW16,SQ8_direct_signed"
            String indexDescription = methodAsMapBuilder.indexDescription;
            if (StringUtils.isNotEmpty(indexDescription)) {
                StringBuilder indexDescriptionBuilder = new StringBuilder();
                indexDescriptionBuilder.append(indexDescription.split(",")[0]);
                indexDescriptionBuilder.append(",");
                indexDescriptionBuilder.append(FAISS_SIGNED_BYTE_SQ);
                methodAsMapBuilder.indexDescription = indexDescriptionBuilder.toString();
            }
        }
        methodAsMapBuilder.indexDescription = prefix + methodAsMapBuilder.indexDescription;
        return methodAsMapBuilder.build();
    }

    @Override
    protected KNNLibrarySearchContext doGetKNNLibrarySearchContext(KNNMethodConfigContext knnMethodConfigContext) {
        CompressionConfig compressionConfig = resolveCompressionConfig(knnMethodConfigContext);
        if (compressionConfig == CompressionConfig.x32) {
            return new FilterKNNLibrarySearchContext(knnLibrarySearchContext) {
                @Override
                public RescoreContext getDefaultRescoreContext(QueryContext ctx) {
                    return RescoreContext.builder().oversampleFactor(2.0f).build();
                }
            };
        }

        if (compressionConfig == CompressionConfig.x16) {
            return new FilterKNNLibrarySearchContext(knnLibrarySearchContext) {
                @Override
                public RescoreContext getDefaultRescoreContext(QueryContext ctx) {
                    return RescoreContext.builder().oversampleFactor(1.5f).build();
                }
            };
        }

        if (compressionConfig == CompressionConfig.x8) {
            return new FilterKNNLibrarySearchContext(knnLibrarySearchContext) {
                @Override
                public RescoreContext getDefaultRescoreContext(QueryContext ctx) {
                    return RescoreContext.builder().oversampleFactor(1.2f).build();
                }
            };
        }

        return knnLibrarySearchContext;
    }

    static CompressionConfig resolveCompressionConfig(KNNMethodConfigContext knnMethodConfigContext) {
        CompressionConfig compressionConfig = knnMethodConfigContext.getCompressionConfig();
        WorkloadModeConfig workloadModeConfig = knnMethodConfigContext.getWorkloadModeConfig();
        if (compressionConfig != CompressionConfig.NOT_CONFIGURED) {
            return compressionConfig;
        }

        if (workloadModeConfig == WorkloadModeConfig.NOT_CONFIGURED) {
            return compressionConfig;
        }

        if (workloadModeConfig == WorkloadModeConfig.IN_MEMORY) {
            return compressionConfig;
        }

        // We default to 32x for ON_DISK
        return CompressionConfig.x32;
    }

    static MethodComponentContext getEncoderMethodComponent(MethodComponentContext methodComponentContext) {
        if (methodComponentContext == null) {
            return null;
        }

        if (methodComponentContext.getParameters().isEmpty()) {
            return null;
        }

        if (!methodComponentContext.getParameters().get().containsKey(METHOD_ENCODER_PARAMETER)) {
            return null;
        }
        Object object = methodComponentContext.getParameters().get().get(METHOD_ENCODER_PARAMETER);
        if (!(object instanceof MethodComponentContext)) {
            return null;
        }
        return (MethodComponentContext) object;
    }
}
