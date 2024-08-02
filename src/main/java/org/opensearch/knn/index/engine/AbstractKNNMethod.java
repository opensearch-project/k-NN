/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.training.VectorSpaceInfo;

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
    protected final EngineSpecificMethodContext engineSpecificMethodContext;

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

        ValidationException methodValidation = methodComponent.validate(knnMethodContext.getMethodComponentContext());
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
    public ValidationException validateWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo) {
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

        ValidationException methodValidation = methodComponent.validateWithData(
            knnMethodContext.getMethodComponentContext(),
            vectorSpaceInfo
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

    @Override
    public Map<String, Object> getAsMap(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameterMap = new HashMap<>(methodComponent.getAsMap(knnMethodContext.getMethodComponentContext()));
        parameterMap.put(KNNConstants.SPACE_TYPE, knnMethodContext.getSpaceType().getValue());
        return parameterMap;
    }

    @Override
    public EngineSpecificMethodContext getEngineSpecificMethodContext() {
        return engineSpecificMethodContext;
    }
}
