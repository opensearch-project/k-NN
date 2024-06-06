/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.training.VectorSpaceInfo;

import java.util.Map;

/**
 * AbstractKNNLibrary implements common functionality shared between libraries
 */
@AllArgsConstructor(access = AccessLevel.PACKAGE)
public abstract class AbstractKNNLibrary implements KNNLibrary {

    protected final Map<String, KNNMethod> methods;
    protected final Map<String, EngineSpecificMethodContext> engineMethods;
    @Getter
    protected final String version;

    @Override
    public KNNMethod getMethod(String methodName) {
        KNNMethod method = methods.get(methodName);
        if (method == null) {
            throw new IllegalArgumentException(String.format("Invalid method name: %s", methodName));
        }
        return method;
    }

    @Override
    public EngineSpecificMethodContext getMethodContext(String methodName) {
        EngineSpecificMethodContext method = engineMethods.get(methodName);
        if (method == null) {
            throw new IllegalArgumentException(String.format("Invalid method name: %s", methodName));
        }
        return method;
    }

    @Override
    public ValidationException validateMethod(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        return getMethod(methodName).validate(knnMethodContext);
    }

    @Override
    public ValidationException validateMethodWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        return getMethod(methodName).validateWithData(knnMethodContext, vectorSpaceInfo);
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        return getMethod(methodName).isTrainingRequired(knnMethodContext);
    }

    @Override
    public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
        KNNMethod knnMethod = methods.get(knnMethodContext.getMethodComponentContext().getName());

        if (knnMethod == null) {
            throw new IllegalArgumentException(
                String.format("Invalid method name: %s", knnMethodContext.getMethodComponentContext().getName())
            );
        }

        return knnMethod.getAsMap(knnMethodContext);
    }
}
