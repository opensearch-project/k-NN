/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.Getter;
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
abstract class NativeLibrary extends AbstractKNNLibrary {
    private final Map<SpaceType, Function<Float, Float>> scoreTranslation;
    @Getter
    private final String extension;
    private final AtomicBoolean initialized;

    /**
     * Constructor for NativeLibrary
     *
     * @param methods map of methods the native library supports
     * @param scoreTranslation Map of translation of space type to scores returned by the library
     * @param version String representation of version of the library
     * @param extension String representing the extension that library files should use
     */
    NativeLibrary(
        Map<String, KNNMethod> methods,
        Map<String, EngineSpecificMethodContext> engineMethods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String version,
        String extension
    ) {
        super(methods, engineMethods, version);
        this.scoreTranslation = scoreTranslation;
        this.extension = extension;
        this.initialized = new AtomicBoolean(false);
    }

    @Override
    public String getCompoundExtension() {
        return getExtension() + KNNConstants.COMPOUND_EXTENSION;
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        if (this.scoreTranslation.containsKey(spaceType)) {
            return this.scoreTranslation.get(spaceType).apply(rawScore);
        }

        return spaceType.scoreTranslation(rawScore);
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();
        return getMethod(methodName).estimateOverheadInKB(knnMethodContext, dimension);
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
