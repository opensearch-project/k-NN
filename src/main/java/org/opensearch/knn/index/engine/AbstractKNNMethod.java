/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.SpaceVectorValidator;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;
import org.opensearch.knn.index.mapper.VectorValidator;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;

/**
 * Abstract class for KNN methods. This class provides the common functionality for all KNN methods.
 * It defines the common attributes and methods that all KNN methods should implement.
 */
@AllArgsConstructor
public abstract class AbstractKNNMethod implements KNNMethod {

    protected final MethodComponent methodComponent;
    protected final Set<SpaceType> spaces;
    protected final KNNLibrarySearchContext knnLibrarySearchContext;

    @Override
    public boolean isSpaceTypeSupported(SpaceType space) {
        return spaces.contains(space);
    }

    @Override
    public ValidationException validate(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        List<String> errorMessages = new ArrayList<>();
        if (!isSpaceTypeSupported(knnMethodContext.getSpaceType())) {
            errorMessages.add(
                String.format(
                    Locale.ROOT,
                    "\"%s\" with \"%s\" configuration does not support space type: " + "\"%s\".",
                    this.methodComponent.getName(),
                    knnMethodContext.getKnnEngine().getName().toLowerCase(Locale.ROOT),
                    knnMethodContext.getSpaceType().getValue()
                )
            );
        }

        ValidationException methodValidation = methodComponent.validate(
            knnMethodContext.getMethodComponentContext(),
            knnMethodConfigContext
        );
        if (methodValidation != null) {
            errorMessages.addAll(methodValidation.validationErrors());
        }

        if (errorMessages.isEmpty()) {
            return null;
        }

        ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        return methodComponent.isTrainingRequired(knnMethodContext.getMethodComponentContext());
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        return methodComponent.estimateOverheadInKB(knnMethodContext.getMethodComponentContext(), knnMethodConfigContext.getDimension());
    }

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
        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    protected VectorValidator doGetVectorValidator(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        return new SpaceVectorValidator(knnMethodContext.getSpaceType());
    }

    protected PerDimensionProcessor doGetPerDimensionProcessor(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }

    protected Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> doGetTrainingConfigValidationSetup() {
        return (trainingConfigValidationInput) -> {

            KNNMethodContext knnMethodContext = trainingConfigValidationInput.getKnnMethodContext();
            KNNMethodConfigContext knnMethodConfigContext = trainingConfigValidationInput.getKnnMethodConfigContext();
            Long trainingVectors = trainingConfigValidationInput.getTrainingVectorsCount();

            TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();

            // validate ENCODER_PARAMETER_PQ_M is divisible by vector dimension
            if (knnMethodContext != null && knnMethodConfigContext != null) {
                if (knnMethodContext.getMethodComponentContext().getParameters().containsKey(ENCODER_PARAMETER_PQ_M)
                    && knnMethodConfigContext.getDimension() % (Integer) knnMethodContext.getMethodComponentContext()
                        .getParameters()
                        .get(ENCODER_PARAMETER_PQ_M) != 0) {
                    builder.valid(false);
                    return builder.build();
                } else {
                    builder.valid(true);
                }
            }

            // validate number of training points should be greater than minimum clustering criteria defined in faiss
            if (knnMethodContext != null && trainingVectors != null) {
                long minTrainingVectorCount = 1000;

                MethodComponentContext encoderContext = (MethodComponentContext) knnMethodContext.getMethodComponentContext()
                    .getParameters()
                    .get(METHOD_ENCODER_PARAMETER);

                if (knnMethodContext.getMethodComponentContext().getParameters().containsKey(METHOD_PARAMETER_NLIST)
                    && encoderContext.getParameters().containsKey(ENCODER_PARAMETER_PQ_CODE_SIZE)) {

                    int nlist = ((Integer) knnMethodContext.getMethodComponentContext().getParameters().get(METHOD_PARAMETER_NLIST));
                    int code_size = ((Integer) encoderContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE));
                    minTrainingVectorCount = (long) Math.max(nlist, Math.pow(2, code_size));
                }

                if (trainingVectors < minTrainingVectorCount) {
                    builder.valid(false).minTrainingVectorCount(minTrainingVectorCount);
                    return builder.build();
                } else {
                    builder.valid(true);
                }
            }
            return builder.build();
        };
    }

    protected VectorTransformer getVectorTransformer(SpaceType spaceType) {
        return VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER;
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        KNNLibraryIndexingContext knnLibraryIndexingContext = methodComponent.getKNNLibraryIndexingContext(
            knnMethodContext.getMethodComponentContext(),
            knnMethodConfigContext
        );
        Map<String, Object> parameterMap = knnLibraryIndexingContext.getLibraryParameters();
        parameterMap.put(KNNConstants.SPACE_TYPE, convertUserToMethodSpaceType(knnMethodContext.getSpaceType()).getValue());
        parameterMap.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, knnMethodConfigContext.getVectorDataType().getValue());
        return KNNLibraryIndexingContextImpl.builder()
            .quantizationConfig(knnLibraryIndexingContext.getQuantizationConfig())
            .parameters(parameterMap)
            .vectorValidator(doGetVectorValidator(knnMethodContext, knnMethodConfigContext))
            .perDimensionValidator(doGetPerDimensionValidator(knnMethodContext, knnMethodConfigContext))
            .perDimensionProcessor(doGetPerDimensionProcessor(knnMethodContext, knnMethodConfigContext))
            .vectorTransformer(getVectorTransformer(knnMethodContext.getSpaceType()))
            .trainingConfigValidationSetup(doGetTrainingConfigValidationSetup())
            .build();
    }

    @Override
    public KNNLibrarySearchContext getKNNLibrarySearchContext() {
        return knnLibrarySearchContext;
    }

    /**
     * Converts user defined space type to method space type that is supported by library.
     * The subclass can override this method and returns the appropriate space type that
     * is supported by the library. This is required because, some libraries may not
     * support all the space types supported by OpenSearch, however. this can be achieved by using compatible space type by the library.
     * For example, faiss does not support cosine similarity. However, we can use inner product space type for cosine similarity after normalization.
     * In this case, we can return the inner product space type for cosine similarity.
     *
     * @param spaceType The space type to check for compatibility
     * @return The compatible space type for the given input, returns the same
     *         space type if it's already compatible
     * @see SpaceType
     */
    protected SpaceType convertUserToMethodSpaceType(SpaceType spaceType) {
        return spaceType;
    }
}
