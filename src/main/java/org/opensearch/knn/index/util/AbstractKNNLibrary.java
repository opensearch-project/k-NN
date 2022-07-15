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
    public KNNMethod getMethod(String methodName) {
        return this.methods.get(methodName);
    }

    @Override
    public ValidationException validateMethod(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponent().getName();
        return getMethod(methodName).validate(knnMethodContext);
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        String methodName = knnMethodContext.getMethodComponent().getName();
        return getMethod(methodName).isTrainingRequired(knnMethodContext);
    }

    @Override
    public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
        KNNMethod knnMethod = methods.get(knnMethodContext.getMethodComponent().getName());

        if (knnMethod == null) {
            throw new IllegalArgumentException(String.format("Invalid method name: %s", knnMethodContext.getMethodComponent().getName()));
        }

        return knnMethod.getAsMap(knnMethodContext);
    }
}
