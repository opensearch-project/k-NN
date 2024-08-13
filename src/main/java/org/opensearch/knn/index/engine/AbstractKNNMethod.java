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
import java.util.HashMap;
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
    public ValidationException validate(KNNMethodContext knnMethodContext) {
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
            knnMethodContext.getKnnMethodConfigContext()
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
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
        return methodComponent.estimateOverheadInKB(knnMethodContext.getMethodComponentContext(), dimension);
    }

    protected PerDimensionValidator doGetPerDimensionValidator(KNNMethodContext knnMethodContext) {
        VectorDataType vectorDataType = knnMethodContext.getKnnMethodConfigContext()
            .getVectorDataType()
            .orElseThrow(
                () -> new IllegalStateException("Vector data type needs to be set on KNNMethodConfigContext in order to get the processor")
            );

        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }
        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    protected VectorValidator doGetVectorValidator(KNNMethodContext knnMethodContext) {
        return new SpaceVectorValidator(knnMethodContext.getSpaceType());
    }

    protected PerDimensionProcessor doGetPerDimensionProcessor(KNNMethodContext knnMethodContext) {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameterMap = new HashMap<>(
            methodComponent.getAsMap(knnMethodContext.getMethodComponentContext(), knnMethodContext.getKnnMethodConfigContext())
        );
        parameterMap.put(KNNConstants.SPACE_TYPE, knnMethodContext.getSpaceType().getValue());
        return KNNLibraryIndexingContextImpl.builder()
            .parameters(parameterMap)
            .vectorValidator(doGetVectorValidator(knnMethodContext))
            .perDimensionValidator(doGetPerDimensionValidator(knnMethodContext))
            .perDimensionProcessor(doGetPerDimensionProcessor(knnMethodContext))
            .build();
    }

    @Override
    public KNNLibrarySearchContext getKNNLibrarySearchContext() {
        return knnLibrarySearchContext;
    }
}
