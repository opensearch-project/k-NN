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
