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


package com.amazon.opendistroforelasticsearch.knn.index.util;

import com.amazon.opendistroforelasticsearch.knn.index.KNNMethod;
import com.amazon.opendistroforelasticsearch.knn.index.KNNMethodContext;
import com.amazon.opendistroforelasticsearch.knn.index.MethodComponent;
import com.amazon.opendistroforelasticsearch.knn.index.Parameter;
import com.amazon.opendistroforelasticsearch.knn.index.SpaceType;
import com.google.common.collect.ImmutableMap;
import org.opensearch.common.ValidationException;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.METHOD_HNSW;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * KNNLibrary is an interface that helps the plugin communicate with k-NN libraries
 */
public interface KNNLibrary {

    /**
     * Gets the library's latest build version
     *
     * @return the string representing the library's latest build version
     */
    String getLatestBuildVersion();

    /**
     * Gets the library's latest version
     *
     * @return the string representing the library's latest version
     */
    String getLatestLibVersion();

    /**
     * Gets the extension that files written with this library should have
     *
     * @return extension
     */
    String getExtension();

    /**
     * Gets the compound extension that files written with this library should have
     *
     * @return compound extension
     */
    String getCompoundExtension();

    /**
     * Gets a particular KNN method that the library supports. This should throw an exception if the method is not
     * supported by the library.
     *
     * @param methodName name of the method to be looked up
     * @return KNNMethod in the library corresponding to the method name
     */
    KNNMethod getMethod(String methodName);

    /**
     * Generate the Lucene score from the rawScore returned by the library. With k-NN, often times the library
     * will return a score where the lower the score, the better the result. This is the opposite of how Lucene scores
     * documents.
     *
     * @param rawScore returned by the library
     * @param spaceType spaceType used to compute the score
     * @return Lucene score for the rawScore
     */
    float score(float rawScore, SpaceType spaceType);

    /**
     * Validate the knnMethodContext for the given library. A ValidationException should be thrown if the method is
     * deemed invalid.
     *
     * @param knnMethodContext to be validated
     */
    void validateMethod(KNNMethodContext knnMethodContext);

    /**
     * Abstract implementation of KNNLibrary. It contains several default methods and fields that
     * are common across different underlying libraries.
     */
    abstract class NativeLibrary implements KNNLibrary {
        protected Map<String, KNNMethod> methods;
        private Map<SpaceType, Function<Float, Float>> scoreTranslation;
        private String latestLibraryBuildVersion;
        private String latestLibraryVersion;
        private String extension;

        /**
         * Constructor for NativeLibrary
         *
         * @param methods map of methods the native library supports
         * @param scoreTranslation Map of translation of space type to scores returned by the library
         * @param latestLibraryBuildVersion String representation of latest build version of the library
         * @param latestLibraryVersion String representation of latest version of the library
         * @param extension String representing the extension that library files should use
         */
        public NativeLibrary(Map<String, KNNMethod> methods, Map<SpaceType, Function<Float, Float>> scoreTranslation,
                             String latestLibraryBuildVersion, String latestLibraryVersion, String extension)
        {
            this.methods = methods;
            this.scoreTranslation = scoreTranslation;
            this.latestLibraryBuildVersion = latestLibraryBuildVersion;
            this.latestLibraryVersion = latestLibraryVersion;
            this.extension = extension;
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
            return getExtension() + "c";
        }

        @Override
        public KNNMethod getMethod(String methodName) {
            if (!methods.containsKey(methodName)) {
                throw new IllegalArgumentException("Invalid method name: " + methodName);
            }
            return methods.get(methodName);
        }

        @Override
        public float score(float rawScore, SpaceType spaceType) {
            if (this.scoreTranslation.containsKey(spaceType)) {
                return this.scoreTranslation.get(spaceType).apply(rawScore);
            }

            return spaceType.scoreTranslation(rawScore);
        }

        @Override
        public void validateMethod(KNNMethodContext knnMethodContext) {
            String methodName = knnMethodContext.getMethodComponent().getName();
            if (!methods.containsKey(methodName)) {
                throw new ValidationException();
            }

            KNNMethod knnMethod = methods.get(methodName);
            knnMethod.validate(knnMethodContext);
        }
    }

    /**
     * Implements NativeLibrary for the nmslib native library
     */
    class Nmslib extends NativeLibrary {
        // ======================================
        // Constants pertaining to nmslib library
        // ======================================
        public final static String HNSW_LIB_NAME = "hnsw";
        public final static String EXTENSION = ".hnsw";

        public final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
                METHOD_HNSW,
                KNNMethod.Builder.builder(
                        MethodComponent.Builder.builder(HNSW_LIB_NAME)
                                .addParameter(METHOD_PARAMETER_M, new Parameter.IntegerParameter(16,
                                        v -> v > 0))
                                .addParameter(METHOD_PARAMETER_EF_CONSTRUCTION, new Parameter.IntegerParameter(512,
                                        v -> v > 0))
                                .build())
                        .addSpaces(SpaceType.L2, SpaceType.L1, SpaceType.LINF, SpaceType.COSINESIMIL,
                                SpaceType.INNER_PRODUCT)
                        .build()
        );

        public final static Map<SpaceType, Function<Float, Float>> SCORE_TRANSLATIONS = Collections.emptyMap();

        public final static Nmslib INSTANCE = new Nmslib(METHODS, SCORE_TRANSLATIONS,
                NmsLibVersion.LATEST.getBuildVersion(), NmsLibVersion.LATEST.indexLibraryVersion(), EXTENSION);

        /**
         * Constructor for Nmslib
         *
         * @param methods Set of methods the native library supports
         * @param scoreTranslation Map of translation of space type to scores returned by the library
         * @param latestLibraryBuildVersion String representation of latest build version of the library
         * @param latestLibraryVersion String representation of latest version of the library
         * @param extension String representing the extension that library files should use
         */
        private Nmslib(Map<String, KNNMethod> methods, Map<SpaceType, Function<Float, Float>> scoreTranslation,
                       String latestLibraryBuildVersion, String latestLibraryVersion, String extension) {
            super(methods, scoreTranslation, latestLibraryBuildVersion, latestLibraryVersion, extension);
        }
    }
}
