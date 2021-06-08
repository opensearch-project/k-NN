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

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.SpaceType;
import com.google.common.collect.ImmutableMap;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

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
     * Generate method as map that can be used to configure the knn index
     *
     * @param knnMethodContext to generate parameter map from
     * @return parameter map
     */
    Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext);

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
        public void validateMethod(KNNMethodContext knnMethodContext) {
            String methodName = knnMethodContext.getMethodComponent().getName();
            getMethod(methodName).validate(knnMethodContext);
        }

        @Override
        public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
            KNNMethod knnMethod = methods.get(knnMethodContext.getMethodComponent().getName());
            return knnMethod.getAsMap(knnMethodContext);
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
                                .addParameter(METHOD_PARAMETER_M, new Parameter.IntegerParameter(
                                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0))
                                .addParameter(METHOD_PARAMETER_EF_CONSTRUCTION, new Parameter.IntegerParameter(
                                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION, v -> v > 0))
                                .build())
                        .addSpaces(SpaceType.L2, SpaceType.L1, SpaceType.LINF, SpaceType.COSINESIMIL,
                                SpaceType.INNER_PRODUCT)
                        .build()
        );


        public final static Nmslib INSTANCE = new Nmslib(METHODS, Collections.emptyMap(),
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

    /**
     * Implements NativeLibrary for the faiss native library
     */
    class Faiss extends NativeLibrary {

        // Map that overrides OpenSearch score translation by space type of scores returned by faiss
        public final static Map<SpaceType, Function<Float, Float>> SCORE_TRANSLATIONS = ImmutableMap.of(
                SpaceType.INNER_PRODUCT, rawScore -> SpaceType.INNER_PRODUCT.scoreTranslation(-1*rawScore)
        );

        // Define encoders supported by faiss
        public final static MethodComponentContext ENCODER_DEFAULT = new MethodComponentContext(
                KNNConstants.ENCODER_FLAT, Collections.emptyMap());

        public final static Map<String, MethodComponent> encoderComponents = ImmutableMap.of(
                        KNNConstants.ENCODER_FLAT, MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
                        .setMapGenerator(((methodComponent, methodComponentContext) ->
                                MethodAsMapBuilder.builder(KNNConstants.ENCODER_FLAT, methodComponent, methodComponentContext.getParameters())
                                        .build()))
                        .build());

        // Define methods supported by faiss
        public final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
                METHOD_HNSW, KNNMethod.Builder.builder(MethodComponent.Builder.builder("HNSW")
                        .addParameter(METHOD_PARAMETER_M,
                                new Parameter.IntegerParameter(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0))
                        .addParameter(METHOD_PARAMETER_EF_CONSTRUCTION,
                                new Parameter.IntegerParameter(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                                        v -> v > 0))
                        .addParameter(METHOD_ENCODER_PARAMETER,
                                new Parameter.MethodComponentContextParameter(ENCODER_DEFAULT, encoderComponents))
                        .setMapGenerator(((methodComponent, methodComponentContext) ->
                                MethodAsMapBuilder.builder("HNSW", methodComponent, methodComponentContext.getParameters())
                                        .addParameter(METHOD_PARAMETER_M, "", "")
                                        .addParameter(METHOD_ENCODER_PARAMETER, ",", "")
                                        .build()))
                        .build())
                        .addSpaces(SpaceType.L2, SpaceType.INNER_PRODUCT).build()
        );

        public final static Faiss INSTANCE = new Faiss(METHODS, SCORE_TRANSLATIONS,
                FaissVersion.LATEST.getBuildVersion(), FaissVersion.LATEST.indexLibraryVersion(),
                KNNConstants.FAISS_EXTENSION);

        /**
         * Constructor for Faiss
         *
         * @param methods                   map of methods the native library supports
         * @param scoreTranslation          Map of translation of space type to scores returned by the library
         * @param latestLibraryBuildVersion String representation of latest build version of the library
         * @param latestLibraryVersion      String representation of latest version of the library
         * @param extension                 String representing the extension that library files should use
         */
        private Faiss(Map<String, KNNMethod> methods, Map<SpaceType,
                Function<Float, Float>> scoreTranslation, String latestLibraryBuildVersion, String latestLibraryVersion,
                      String extension) {
            super(methods, scoreTranslation, latestLibraryBuildVersion, latestLibraryVersion, extension);
        }

        //TODO: Clean a lot
        private static class MethodAsMapBuilder {
            String methodDescription;
            Map<String, Object> parameterMap;
            MethodComponent methodComponent;

            MethodAsMapBuilder(String baseDescription, MethodComponent methodComponent, Map<String, Object> initialParameterMap) {
                this.methodDescription = baseDescription;
                this.methodComponent = methodComponent;
                this.parameterMap = initialParameterMap;
            }

            MethodAsMapBuilder addParameter(String parameterName, String prefix, String suffix) {
                methodDescription += prefix;

                Object parameter = methodComponent.getParameters().get(parameterName);
                Object value = parameterMap.get(parameterName);

                if (parameter instanceof Parameter.MethodComponentContextParameter) {
                    // Some ugly parsing here
                    MethodComponentContext methodComponentContext = (MethodComponentContext) value;
                    MethodComponent methodComponent = ((Parameter.MethodComponentContextParameter) parameter).getMethodComponent(methodComponentContext.getName());

                    Map<String, Object> subMethodComponentAsMap = methodComponent.getAsMap(methodComponentContext);
                    methodDescription += subMethodComponentAsMap.get(KNNConstants.KNN_METHOD);
                    subMethodComponentAsMap.remove(KNNConstants.KNN_METHOD);

                    // We replace parameterName with the map that contains only parameters that are not included in
                    // the method description
                    parameterMap.put(parameterName, subMethodComponentAsMap);
                } else {
                    // Just add the value to the method description and remove from map
                    methodDescription += value;
                    parameterMap.remove(parameterName);
                }

                methodDescription += suffix;
                return this;
            }

            Map<String, Object> build() {
                parameterMap.put(KNNConstants.KNN_METHOD, methodDescription);
                return parameterMap;
            }

            static MethodAsMapBuilder builder(String baseDescription, MethodComponent methodComponent,
                                              Map<String, Object> initialParameterMap) {
                return new MethodAsMapBuilder(baseDescription, methodComponent, initialParameterMap);
            }
        }

        /**
         * Enum containing information about faiss versioning
         */
        private enum FaissVersion {

            /**
             * Latest available nmslib version
             */
            V165("165"){
                @Override
                public String indexLibraryVersion() {
                    return KNNConstants.JNI_LIBRARY_NAME;
                }
            };

            static final FaissVersion LATEST = V165;

            String buildVersion;

            FaissVersion(String buildVersion) {
                this.buildVersion = buildVersion;
            }

            /**
             * Faiss version used by the KNN codec
             * @return library name
             */
            abstract String indexLibraryVersion();

            String getBuildVersion() { return buildVersion; }
        }
    }
}
