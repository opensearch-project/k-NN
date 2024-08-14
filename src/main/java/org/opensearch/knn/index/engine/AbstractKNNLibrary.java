/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.VectorDataType;

import java.util.Locale;
import java.util.Map;

/**
 * AbstractKNNLibrary implements common functionality shared between libraries
 */
@AllArgsConstructor(access = AccessLevel.PACKAGE)
public abstract class AbstractKNNLibrary implements KNNLibrary {

    protected final Map<String, KNNMethod> methods;
    @Getter
    protected final String version;

    @Override
    public KNNLibrarySearchContext getKNNLibrarySearchContext(String methodName) {
        throwIllegalArgOnNonNull(validateMethodExists(methodName));
        KNNMethod method = methods.get(methodName);
        return method.getKNNLibrarySearchContext();
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        String method = knnMethodContext.getMethodComponentContext().getName();
        throwIllegalArgOnNonNull(validateMethodExists(method));
        KNNMethod knnMethod = methods.get(method);
        return knnMethod.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @Override
    public ValidationException validateMethod(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        ValidationException validationException = null;
        String invalidErrorMessage = validateMethodExists(methodName);
        if (invalidErrorMessage != null) {
            validationException = new ValidationException();
            validationException.addValidationError(invalidErrorMessage);
            return validationException;
        }
        invalidErrorMessage = validateDimension(knnMethodContext, knnMethodConfigContext);
        if (invalidErrorMessage != null) {
            validationException = new ValidationException();
            validationException.addValidationError(invalidErrorMessage);
        }

        validateSpaceType(knnMethodContext, knnMethodConfigContext);
        ValidationException methodValidation = methods.get(methodName).validate(knnMethodContext, knnMethodConfigContext);
        if (methodValidation != null) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationErrors(methodValidation.validationErrors());
        }

        return validationException;
    }

    private void validateSpaceType(final KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodContext == null) {
            return;
        }
        knnMethodContext.getSpaceType().validateVectorDataType(knnMethodConfigContext.getVectorDataType());
    }

    private String validateDimension(final KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodContext == null) {
            return null;
        }
        int dimension = knnMethodConfigContext.getDimension();
        if (dimension > KNNEngine.getMaxDimensionByEngine(knnMethodContext.getKnnEngine())) {
            return String.format(
                Locale.ROOT,
                "Dimension value cannot be greater than %s for vector with engine: %s",
                KNNEngine.getMaxDimensionByEngine(knnMethodContext.getKnnEngine()),
                knnMethodContext.getKnnEngine().getName()
            );
        }

        if (VectorDataType.BINARY == knnMethodConfigContext.getVectorDataType() && dimension % 8 != 0) {
            return "Dimension should be multiply of 8 for binary vector data type";
        }

        return null;
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        throwIllegalArgOnNonNull(validateMethodExists(methodName));
        return methods.get(methodName).isTrainingRequired(knnMethodContext);
    }

    private String validateMethodExists(String methodName) {
        KNNMethod method = methods.get(methodName);
        if (method == null) {
            return String.format("Invalid method name: %s", methodName);
        }
        return null;
    }

    private void throwIllegalArgOnNonNull(String errorMessage) {
        if (errorMessage != null) {
            throw new IllegalArgumentException(errorMessage);
        }
    }
}
