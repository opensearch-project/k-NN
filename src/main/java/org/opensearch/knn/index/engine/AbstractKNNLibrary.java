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
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(KNNMethodConfigContext knnMethodConfigContext) {
        String methodName = resolveMethod(knnMethodConfigContext);
        throwIllegalArgOnNonNull(validateMethodExists(methodName));
        KNNMethod knnMethod = methods.get(methodName);
        return knnMethod.getKNNLibraryIndexingContext(knnMethodConfigContext);
    }

    @Override
    public ValidationException validateMethod(KNNMethodConfigContext knnMethodConfigContext) {
        String methodName = resolveMethod(knnMethodConfigContext);
        ValidationException validationException = null;
        String invalidErrorMessage = validateMethodExists(methodName);
        if (invalidErrorMessage != null) {
            validationException = new ValidationException();
            validationException.addValidationError(invalidErrorMessage);
            return validationException;
        }
        invalidErrorMessage = validateDimension(knnMethodConfigContext);
        if (invalidErrorMessage != null) {
            validationException = new ValidationException();
            validationException.addValidationError(invalidErrorMessage);
        }

        validateSpaceType(knnMethodConfigContext);
        ValidationException methodValidation = methods.get(methodName).validate(knnMethodConfigContext);
        if (methodValidation != null) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationErrors(methodValidation.validationErrors());
        }

        return validationException;
    }

    @Override
    public boolean isTrainingRequired(KNNMethodConfigContext knnMethodConfigContext) {
        String methodName = resolveMethod(knnMethodConfigContext);
        throwIllegalArgOnNonNull(validateMethodExists(methodName));
        return methods.get(methodName).isTrainingRequired(knnMethodConfigContext);
    }

    /**
     * Resolve the method name from the config
     *
     * @param knnMethodConfigContext knnMethodConfigContext
     * @return method name
     */
    protected String resolveMethod(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext.getKnnMethodContext() != null
            && knnMethodConfigContext.getKnnMethodContext().getMethodComponentContext().getName().isPresent()) {
            return knnMethodConfigContext.getKnnMethodContext().getMethodComponentContext().getName().get();
        }
        return doResolveMethod(knnMethodConfigContext);
    }

    protected abstract String doResolveMethod(KNNMethodConfigContext knnMethodConfigContext);

    private void validateSpaceType(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext == null) {
            return;
        }
        KNNMethodContext knnMethodContext = knnMethodConfigContext.getKnnMethodContext();
        if (knnMethodContext == null) {
            return;
        }
        knnMethodConfigContext.getSpaceType().validateVectorDataType(knnMethodConfigContext.getVectorDataType());
    }

    private String validateDimension(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext == null) {
            return null;
        }
        int dimension = knnMethodConfigContext.getDimension();
        KNNEngine knnEngine = knnMethodConfigContext.getKnnEngine();
        if (dimension > KNNEngine.getMaxDimensionByEngine(knnEngine)) {
            return String.format(
                Locale.ROOT,
                "Dimension value cannot be greater than %s for vector with engine: %s",
                KNNEngine.getMaxDimensionByEngine(knnEngine),
                knnEngine.getName()
            );
        }

        if (VectorDataType.BINARY == knnMethodConfigContext.getVectorDataType() && dimension % 8 != 0) {
            return "Dimension should be multiply of 8 for binary vector data type";
        }

        return null;
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
