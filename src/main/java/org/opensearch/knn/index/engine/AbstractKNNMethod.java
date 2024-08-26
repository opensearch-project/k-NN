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
    public ValidationException validate(KNNMethodConfigContext knnMethodConfigContext) {
        List<String> errorMessages = new ArrayList<>();
        SpaceType spaceType = knnMethodConfigContext.getSpaceType();
        if (!isSpaceTypeSupported(spaceType)) {
            errorMessages.add(
                String.format(
                    Locale.ROOT,
                    "\"%s\" with \"%s\" configuration does not support space type: " + "\"%s\".",
                    this.methodComponent.getName(),
                    knnMethodConfigContext.getKnnEngine().getName().toLowerCase(Locale.ROOT),
                    spaceType.getValue()
                )
            );
        }

        MethodComponentContext methodComponentContext = extractUserProvidedMethodComponentContext(knnMethodConfigContext);
        ValidationException methodValidation = methodComponent.validate(methodComponentContext, knnMethodConfigContext);
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
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(KNNMethodConfigContext knnMethodConfigContext) {
        MethodComponentContext resolvedMethodComponentContext = resolveMethodComponentContext(knnMethodConfigContext);
        KNNLibraryIndexingContext knnLibraryIndexingContext = methodComponent.getKNNLibraryIndexingContext(
            resolvedMethodComponentContext,
            knnMethodConfigContext
        );
        Map<String, Object> parameterMap = knnLibraryIndexingContext.getLibraryParameters();
        SpaceType spaceType = knnMethodConfigContext.getSpaceType();
        parameterMap.put(KNNConstants.SPACE_TYPE, spaceType.getValue());
        parameterMap.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, knnMethodConfigContext.getVectorDataType().getValue());
        return KNNLibraryIndexingContextImpl.builder()
            .quantizationConfig(knnLibraryIndexingContext.getQuantizationConfig())
            .parameters(parameterMap)
            .vectorValidator(doGetVectorValidator(knnMethodConfigContext))
            .perDimensionValidator(doGetPerDimensionValidator(knnMethodConfigContext))
            .perDimensionProcessor(doGetPerDimensionProcessor(knnMethodConfigContext))
            .knnLibrarySearchContext(doGetKNNLibrarySearchContext(knnMethodConfigContext))
            .estimateOverheadInKB(knnLibraryIndexingContext.estimateOverheadInKB())
            .build();
    }

    @Override
    public boolean isTrainingRequired(KNNMethodConfigContext knnMethodConfigContext) {
        MethodComponentContext resolvedMethodComponentContext = resolveMethodComponentContext(knnMethodConfigContext);
        return methodComponent.isTrainingRequired(resolvedMethodComponentContext, knnMethodConfigContext);
    }

    protected MethodComponentContext extractUserProvidedMethodComponentContext(KNNMethodConfigContext knnMethodConfigContext) {
        KNNMethodContext knnMethodContext = knnMethodConfigContext.getKnnMethodContext();
        if (knnMethodContext == null) {
            return null;
        }
        return knnMethodContext.getMethodComponentContext();
    }

    protected MethodComponentContext resolveMethodComponentContext(KNNMethodConfigContext knnMethodConfigContext) {
        MethodComponentContext providedMethodComponentContext = extractUserProvidedMethodComponentContext(knnMethodConfigContext);
        return methodComponent.resolveMethodComponentContext(knnMethodConfigContext, providedMethodComponentContext);
    }

    protected PerDimensionValidator doGetPerDimensionValidator(KNNMethodConfigContext knnMethodConfigContext) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();

        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }
        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    protected VectorValidator doGetVectorValidator(KNNMethodConfigContext knnMethodConfigContext) {
        SpaceType spaceType = knnMethodConfigContext.getSpaceType();
        return new SpaceVectorValidator(spaceType);
    }

    protected PerDimensionProcessor doGetPerDimensionProcessor(KNNMethodConfigContext knnMethodConfigContext) {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }

    protected KNNLibrarySearchContext doGetKNNLibrarySearchContext(KNNMethodConfigContext knnMethodConfigContext) {
        return knnLibrarySearchContext;
    }

    private boolean isSpaceTypeSupported(SpaceType space) {
        return spaces.contains(space);
    }
}
