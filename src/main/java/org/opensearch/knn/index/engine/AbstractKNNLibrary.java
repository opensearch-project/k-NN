/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.training.VectorSpaceInfo;

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
    public EngineSpecificMethodContext getMethodContext(String methodName) {
        validateMethodExists(methodName);
        KNNMethod method = methods.get(methodName);
        return method.getEngineSpecificMethodContext();
    }

    @Override
    public ValidationException validateMethod(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        validateMethodExists(methodName);
        return methods.get(methodName).validate(knnMethodContext);
    }

    @Override
    public ValidationException validateMethodWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        validateMethodExists(methodName);
        return methods.get(methodName).validateWithData(knnMethodContext, vectorSpaceInfo);
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        validateMethodExists(methodName);
        return methods.get(methodName).isTrainingRequired(knnMethodContext);
    }

    @Override
    public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
        String method = knnMethodContext.getMethodComponentContext().getName();
        validateMethodExists(method);
        KNNMethod knnMethod = methods.get(method);
        return knnMethod.getAsMap(knnMethodContext);
    }

    private void validateMethodExists(String methodName) {
        KNNMethod method = methods.get(methodName);
        if (method == null) {
            throw new IllegalArgumentException(String.format("Invalid method name: %s", methodName));
        }
    }
}
