/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.util;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Abstract implementation of KNNLibrary. It contains several default methods and fields that
 * are common across different underlying libraries.
 */
abstract class NativeLibrary implements KNNLibrary {
    protected Map<String, KNNMethod> methods;
    private final Map<SpaceType, Function<Float, Float>> scoreTranslation;
    private final String latestLibraryBuildVersion;
    private final String latestLibraryVersion;
    private final String extension;
    private final AtomicBoolean initialized;

    /**
     * Constructor for NativeLibrary
     *
     * @param methods map of methods the native library supports
     * @param scoreTranslation Map of translation of space type to scores returned by the library
     * @param latestLibraryBuildVersion String representation of latest build version of the library
     * @param latestLibraryVersion String representation of latest version of the library
     * @param extension String representing the extension that library files should use
     */
    public NativeLibrary(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String latestLibraryBuildVersion,
        String latestLibraryVersion,
        String extension
    ) {
        this.methods = methods;
        this.scoreTranslation = scoreTranslation;
        this.latestLibraryBuildVersion = latestLibraryBuildVersion;
        this.latestLibraryVersion = latestLibraryVersion;
        this.extension = extension;
        this.initialized = new AtomicBoolean(false);
    }

    @Override
    public String getLatestBuildVersion() {
        return this.latestLibraryBuildVersion;
    }

    @Override
    public String getLatestLibVersion() {
        return this.latestLibraryVersion;
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
        throw new IllegalArgumentException("Invalid method name: " + methodName);
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
            throw new IllegalArgumentException("Invalid method name: " + knnMethodContext.getMethodComponent().getName());
        }

        return knnMethod.getAsMap(knnMethodContext);
    }

    @Override
    public Boolean isInitialized() {
        return initialized.get();
    }

    @Override
    public void setInitialized(Boolean isInitialized) {
        initialized.set(isInitialized);
    }
}
