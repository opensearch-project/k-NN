/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.engine.faiss.AbstractFaissMethod;
import org.opensearch.knn.index.engine.faiss.FaissFlatEncoder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.sandbox.svs.SVSConstants.DEFAULT_CONSTRUCTION_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LEANVEC;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_VAMANA_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_ALPHA;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_DEGREE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_SVS_VAMANA;

/**
 * SVS Vamana method: graph-based approximate search using the Vamana algorithm (Subramanya et al.).
 */
public class FaissSVSVamanaMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = Set.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);

    public final static Map<String, Encoder> SUPPORTED_ENCODERS = Map.of(
        ENCODER_FLAT,
        new FaissFlatEncoder(),
        ENCODER_SQ,
        new FaissSVSSQEncoder(),
        FAISS_SVS_ENCODER_LVQ,
        new FaissSVSLVQEncoder(),
        FAISS_SVS_ENCODER_LEANVEC,
        new FaissSVSLeanVecEncoder()
    );

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap());

    public final static MethodComponent METHOD_COMPONENT = initMethodComponent();

    public FaissSVSVamanaMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new FaissSVSVamanaSearchContext());
    }

    /**
     * Mirrors the {@code AbstractKNNMethod} default but builds the {@link ResolvedIndexSpec} itself: the default
     * resolves encoder names through the closed {@code Encoder.EncoderType} enum, which rejects lvq/leanvec.
     */
    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        KNNLibraryIndexingContext componentContext = methodComponent.getKNNLibraryIndexingContext(
            knnMethodContext.getMethodComponentContext(),
            knnMethodConfigContext
        );
        Map<String, Object> parameterMap = componentContext.getLibraryParameters();
        parameterMap.put(SPACE_TYPE, convertUserToMethodSpaceType(knnMethodContext.getSpaceType()).getValue());
        parameterMap.put(VECTOR_DATA_TYPE_FIELD, knnMethodConfigContext.getVectorDataType().getValue());
        return KNNLibraryIndexingContextImpl.builder()
            .quantizationConfig(componentContext.getQuantizationConfig())
            .parameters(parameterMap)
            .vectorValidator(doGetVectorValidator(knnMethodContext, knnMethodConfigContext))
            .perDimensionValidator(doGetPerDimensionValidator(knnMethodContext, knnMethodConfigContext))
            .perDimensionProcessor(doGetPerDimensionProcessor(knnMethodContext, knnMethodConfigContext))
            .vectorTransformer(getVectorTransformer(knnMethodContext.getSpaceType()))
            .trainingConfigValidationSetup(doGetTrainingConfigValidationSetup())
            .resolvedSpec(buildSvsResolvedSpec(knnMethodContext, knnMethodConfigContext))
            .build();
    }

    static ResolvedIndexSpec buildSvsResolvedSpec(KNNMethodContext methodContext, KNNMethodConfigContext configContext) {
        Encoder.EncoderType encoderType = Encoder.EncoderType.FLAT;
        Encoder.QuantizationBits quantizationBits = Encoder.QuantizationBits.FULL_PRECISION;

        Map<String, Object> methodParams = methodContext.getMethodComponentContext().getParameters();
        if (methodParams != null && methodParams.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext encoderCtx) {
            String encoderName = encoderCtx.getName();
            if (ENCODER_FLAT.equals(encoderName)) {
                encoderType = Encoder.EncoderType.FLAT;
                quantizationBits = Encoder.QuantizationBits.FULL_PRECISION;
            } else if (ENCODER_SQ.equals(encoderName)) {
                encoderType = Encoder.EncoderType.SQ;
                Object sqType = encoderCtx.getParameters().get(SVSConstants.FAISS_SVS_SQ_TYPE);
                quantizationBits = SVSConstants.FAISS_SVS_SQ_TYPE_SQ8.equals(sqType)
                    ? Encoder.QuantizationBits.SEVEN
                    : Encoder.QuantizationBits.SIXTEEN;
            } else {
                // lvq / leanvec: closest fit in the closed enum.
                encoderType = Encoder.EncoderType.SQ;
                quantizationBits = Encoder.QuantizationBits.SEVEN;
            }
        }

        Integer dimension = configContext.getDimension();
        return ResolvedIndexSpec.builder()
            .engine(methodContext.getKnnEngine())
            .methodName(methodContext.getMethodComponentContext().getName())
            .encoderType(encoderType)
            .quantizationBits(quantizationBits)
            .compressionLevel(configContext.getCompressionLevel())
            .mode(
                Mode.isConfigured(configContext.getMode())
                    ? configContext.getMode()
                    : KNNMethodConfigContext.deriveMode(configContext.getUserConfiguredCompressionLevel())
            )
            .vectorDataType(configContext.getVectorDataType())
            .dimension(dimension != null ? dimension : 0)
            .indexVersionCreated(configContext.getVersionCreated())
            .memoryOptimizedEligibleOverride(Boolean.FALSE)
            .build();
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_VAMANA)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_DEGREE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_DEGREE, 64, (v, context) -> v > 0 && v <= 256)
            )
            .addParameter(
                METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE,
                    DEFAULT_CONSTRUCTION_WINDOW_SIZE,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, 10, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, 10, (v, context) -> v > 0)
            )
            // null default: the runtime applies its metric-dependent alpha.
            .addParameter(METHOD_PARAMETER_ALPHA, new Parameter.DoubleParameter(METHOD_PARAMETER_ALPHA, null, (v, context) -> v > 0))
            .addParameter(
                METHOD_ENCODER_PARAMETER,
                new Parameter.MethodComponentContextParameter(
                    METHOD_ENCODER_PARAMETER,
                    DEFAULT_ENCODER_CONTEXT,
                    SUPPORTED_ENCODERS.values().stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent))
                )
            )
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                SvsMethodAsMapBuilder methodAsMapBuilder = SvsMethodAsMapBuilder.builder(
                    FAISS_SVS_VAMANA_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );
                methodAsMapBuilder.addParameter(METHOD_PARAMETER_DEGREE, "", "");

                methodAsMapBuilder.addParameter(METHOD_ENCODER_PARAMETER, ",", "");
                methodAsMapBuilder.dropTrailingDescriptionToken(FAISS_FLAT_DESCRIPTION);

                return methodAsMapBuilder.build();
            }))
            .build();
    }
}
