/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Abstract implementation of KNNLibrary. It contains several default methods and fields that
 * are common across different underlying libraries.
 */
abstract class NativeLibrary implements KNNLibrary {
    protected Map<String, KNNMethod> methods;
    private final Map<SpaceType, Function<Float, Float>> scoreTranslation;
    private final String currentVersion;
    private final String extension;
    private final AtomicBoolean initialized;

    /**
     * Constructor for NativeLibrary
     *
     * @param methods map of methods the native library supports
     * @param scoreTranslation Map of translation of space type to scores returned by the library
     * @param currentVersion String representation of current version of the library
     * @param extension String representing the extension that library files should use
     */
    NativeLibrary(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String currentVersion,
        String extension
    ) {
        this.methods = methods;
        this.scoreTranslation = scoreTranslation;
        this.currentVersion = currentVersion;
        this.extension = extension;
        this.initialized = new AtomicBoolean(false);
    }

    @Override
    public String getVersion() {
        return this.currentVersion;
    }

    @Override
    public String getExtension() {
        return this.extension;
    }

    @Override
    public String getCompoundExtension() {
        return getExtension() + KNNConstants.COMPOUND_EXTENSION;
    }

    @Override
    public KNNMethod getMethod(String methodName) {
        KNNMethod method = methods.get(methodName);
        if (method != null) {
            return method;
        }
        throw new IllegalArgumentException(String.format("Invalid method name: %s", methodName));
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        if (this.scoreTranslation.containsKey(spaceType)) {
            return this.scoreTranslation.get(spaceType).apply(rawScore);
        }

        return spaceType.scoreTranslation(rawScore);
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
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
        String methodName = knnMethodContext.getMethodComponent().getName();
        return getMethod(methodName).estimateOverheadInKB(knnMethodContext, dimension);
    }

    @Override
    public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
        KNNMethod knnMethod = methods.get(knnMethodContext.getMethodComponent().getName());

        if (knnMethod == null) {
            throw new IllegalArgumentException(String.format("Invalid method name: %s", knnMethodContext.getMethodComponent().getName()));
        }

        return knnMethod.getAsMap(knnMethodContext);
    }

    @Override
    public Boolean isInitialized() {
        return initialized.get();
    }

    @Override
    public void setInitialized(Boolean isInitialized) {
        Objects.requireNonNull(isInitialized, "isInitialized must not be null");
        initialized.set(isInitialized);
    }
}
