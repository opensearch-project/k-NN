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
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT;
import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_IVF_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_LIMIT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_LIMIT;

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
     * Returns whether training is required or not from knnMethodContext for the given library.
     *
     * @param knnMethodContext methodContext
     * @return true if training is required; false otherwise
     */
    boolean isTrainingRequired(KNNMethodContext knnMethodContext);

    /**
     * Estimate overhead of KNNMethodContext in Kilobytes.
     *
     * @param knnMethodContext to estimate size for
     * @param dimension to estimate size for
     * @return size overhead estimate in KB
     */
    int estimateOverheadInKB (KNNMethodContext knnMethodContext, int dimension);

    /**
     * Generate method as map that can be used to configure the knn index from the jni
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
                throw new IllegalArgumentException("Invalid method name: "
                        + knnMethodContext.getMethodComponent().getName());
            }

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
                Version.LATEST.getBuildVersion(), Version.LATEST.indexLibraryVersion(), EXTENSION);

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

        public enum Version {
            /**
             * Latest available nmslib version
             */
            V2011("2011"){
                @Override
                public String indexLibraryVersion() {
                    return KNNConstants.JNI_LIBRARY_NAME;
                }
            };

            public static final Version LATEST = V2011;

            private String buildVersion;

            Version(String buildVersion) {
                this.buildVersion = buildVersion;
            }

            /**
             * NMS library version used by the KNN codec
             * @return nmslib name
             */
            public abstract String indexLibraryVersion();

            public String getBuildVersion() { return buildVersion; }
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

        //TODO: To think about in future: for PQ, if dimension is not divisible by code count, PQ will fail. Right now,
        // we do not have a way to base validation off of dimension. Failure will happen during training in JNI.
        public final static Map<String, MethodComponent> encoderComponents = ImmutableMap.of(
                KNNConstants.ENCODER_FLAT, MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
                        .setMapGenerator(((methodComponent, methodComponentContext) ->
                                MethodAsMapBuilder.builder(KNNConstants.FAISS_FLAT_DESCRIPTION, methodComponent,
                                        methodComponentContext).build())).build(),
                KNNConstants.ENCODER_PQ, MethodComponent.Builder.builder(KNNConstants.ENCODER_PQ)
                        .addParameter(ENCODER_PARAMETER_PQ_CODE_COUNT,
                                new Parameter.IntegerParameter(ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT, v -> v > 0
                                        && v < ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT))
                        .addParameter(ENCODER_PARAMETER_PQ_CODE_SIZE,
                                new Parameter.IntegerParameter(ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT, v -> v > 0
                                        && v < ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT))
                        .setRequiresTraining(true)
                        .setMapGenerator(((methodComponent, methodComponentContext) ->
                                MethodAsMapBuilder.builder(FAISS_PQ_DESCRIPTION, methodComponent, methodComponentContext)
                                        .addParameter(ENCODER_PARAMETER_PQ_CODE_COUNT, "", "")
                                        .addParameter(ENCODER_PARAMETER_PQ_CODE_SIZE, "x", "")
                                        .build()))
                        .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                            // Size estimate formula: (4 * d * 2^code_size) / 1024 + 1

                            // Get value of code size passed in by user
                            Object codeSizeObject = methodComponentContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);

                            // If not specified, get default value of code size
                            if (codeSizeObject == null) {
                                Object codeSizeParameter = methodComponent.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);
                                if (codeSizeParameter == null) {
                                    throw new IllegalStateException(ENCODER_PARAMETER_PQ_CODE_SIZE + " is not a valid " +
                                            " parameter. This is a bug.");
                                }

                                codeSizeObject = ((Parameter<?>) codeSizeParameter).getDefaultValue();
                            }

                            if (!(codeSizeObject instanceof Integer)) {
                                throw new IllegalStateException(ENCODER_PARAMETER_PQ_CODE_SIZE + " must be " +
                                        "an integer.");
                            }

                            int codeSize = (Integer) codeSizeObject;
                            return ((4L *  (1 << codeSize) * dimension) / BYTES_PER_KILOBYTES) + 1;
                        })
                        .build()
        );

        // Define methods supported by faiss
        public final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
                METHOD_HNSW, KNNMethod.Builder.builder(MethodComponent.Builder.builder(METHOD_HNSW)
                        .addParameter(METHOD_PARAMETER_M,
                                new Parameter.IntegerParameter(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0))
                        .addParameter(METHOD_PARAMETER_EF_CONSTRUCTION,
                                new Parameter.IntegerParameter(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                                        v -> v > 0))
                        .addParameter(METHOD_PARAMETER_EF_SEARCH,
                                new Parameter.IntegerParameter(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                                        v -> v > 0))
                        .addParameter(METHOD_ENCODER_PARAMETER,
                                new Parameter.MethodComponentContextParameter(ENCODER_DEFAULT, encoderComponents))
                        .setMapGenerator(((methodComponent, methodComponentContext) ->
                                MethodAsMapBuilder.builder(FAISS_HNSW_DESCRIPTION, methodComponent, methodComponentContext)
                                        .addParameter(METHOD_PARAMETER_M, "", "")
                                        .addParameter(METHOD_ENCODER_PARAMETER, ",", "")
                                        .build()))
                        .build())
                        .addSpaces(SpaceType.L2, SpaceType.INNER_PRODUCT).build(),
                METHOD_IVF, KNNMethod.Builder.builder(MethodComponent.Builder.builder(METHOD_IVF)
                        .addParameter(METHOD_PARAMETER_NPROBES,
                                new Parameter.IntegerParameter(METHOD_PARAMETER_NPROBES_DEFAULT, v -> v > 0 && v < METHOD_PARAMETER_NPROBES_LIMIT))
                        .addParameter(METHOD_PARAMETER_NLIST,
                                new Parameter.IntegerParameter(METHOD_PARAMETER_NLIST_DEFAULT, v -> v > 0 && v < METHOD_PARAMETER_NLIST_LIMIT))
                        .addParameter(METHOD_ENCODER_PARAMETER,
                                new Parameter.MethodComponentContextParameter(ENCODER_DEFAULT, encoderComponents))
                        .setRequiresTraining(true)
                        .setMapGenerator(((methodComponent, methodComponentContext) ->
                                MethodAsMapBuilder.builder(FAISS_IVF_DESCRIPTION, methodComponent, methodComponentContext)
                                        .addParameter(METHOD_PARAMETER_NLIST, "", "")
                                        .addParameter(METHOD_ENCODER_PARAMETER, ",", "")
                                        .build()))
                        .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                            // Size estimate formula: (4 * nlists * d) / 1024 + 1

                            // Get value of nlists passed in by user
                            Object nlistObject = methodComponentContext.getParameters().get(METHOD_PARAMETER_NLIST);

                            // If not specified, get default value of nlist
                            if (nlistObject == null) {
                                Object nlistParameter = methodComponent.getParameters().get(METHOD_PARAMETER_NLIST);
                                if (nlistParameter == null) {
                                    throw new IllegalStateException(METHOD_PARAMETER_NLIST + " is not a valid " +
                                            " parameter. This is a bug.");
                                }

                                nlistObject = ((Parameter<?>) nlistParameter).getDefaultValue();
                            }

                            if (!(nlistObject instanceof Integer)) {
                                throw new IllegalStateException(METHOD_PARAMETER_NLIST + " must be " +
                                        "an integer.");
                            }

                            int centroids = (Integer) nlistObject;
                            return ((4L *  centroids * dimension) / BYTES_PER_KILOBYTES) + 1;
                        })
                        .build())
                        .addSpaces(SpaceType.L2, SpaceType.INNER_PRODUCT).build()
        );

        public final static Faiss INSTANCE = new Faiss(METHODS, SCORE_TRANSLATIONS,
                Version.LATEST.getBuildVersion(), Version.LATEST.indexLibraryVersion(),
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

        /**
         * MethodAsMap builder is used to create the map that will be passed to the jni to create the faiss index.
         * Faiss's index factory takes an "index description" that it uses to build the index. In this description,
         * some parameters of the index can be configured; others need to be manually set. MethodMap builder creates
         * the index description from a set of parameters and removes them from the map. On build, it sets the index
         * description in the map and returns the processed map
         */
        protected static class MethodAsMapBuilder {
            String indexDescription;
            MethodComponent methodComponent;
            Map<String, Object> methodAsMap;

            /**
             * Constructor
             *
             * @param baseDescription the basic description this component should start with
             * @param methodComponent the method component that maps to this builder
             * @param initialMap the initial parameter map that will be modified
             */
            MethodAsMapBuilder(String baseDescription, MethodComponent methodComponent,
                               Map<String, Object> initialMap) {
                this.indexDescription = baseDescription;
                this.methodComponent = methodComponent;
                this.methodAsMap = initialMap;
            }

            /**
             * Add a parameter that will be used in the index description for the given method component
             *
             * @param parameterName name of the parameter
             * @param prefix to append to the index description before the parameter
             * @param suffix to append to the index description after the parameter
             * @return this builder
             */
            MethodAsMapBuilder addParameter(String parameterName, String prefix, String suffix) {
                indexDescription += prefix;

                Parameter<?> parameter = methodComponent.getParameters().get(parameterName);
                Object value = methodAsMap.containsKey(parameterName) ? methodAsMap.get(parameterName)
                        : parameter.getDefaultValue();

                // Recursion is needed if the parameter is a method component context itself.
                if (parameter instanceof Parameter.MethodComponentContextParameter) {
                    MethodComponentContext subMethodComponentContext = (MethodComponentContext) value;
                    MethodComponent subMethodComponent = ((Parameter.MethodComponentContextParameter) parameter)
                            .getMethodComponent(subMethodComponentContext.getName());

                    Map<String, Object> subMethodAsMap = subMethodComponent.getAsMap(subMethodComponentContext);
                    indexDescription += subMethodAsMap.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER);
                    subMethodAsMap.remove(KNNConstants.INDEX_DESCRIPTION_PARAMETER);

                    // We replace parameterName with the map that contains only parameters that are not included in
                    // the method description
                    methodAsMap.put(parameterName, subMethodAsMap);
                } else {
                    // Just add the value to the method description and remove from map
                    indexDescription += value;
                    methodAsMap.remove(parameterName);
                }

                indexDescription += suffix;
                return this;
            }

            /**
             * Build
             *
             * @return Method as a map
             */
            Map<String, Object> build() {
                methodAsMap.put(KNNConstants.INDEX_DESCRIPTION_PARAMETER, indexDescription);
                return methodAsMap;
            }

            static MethodAsMapBuilder builder(String baseDescription, MethodComponent methodComponent,
                                              MethodComponentContext methodComponentContext) {
                return new MethodAsMapBuilder(baseDescription, methodComponent,
                        new HashMap<>(methodComponentContext.getParameters()));
            }
        }

        /**
         * Enum containing information about faiss versioning
         */
        private enum Version {

            /**
             * Latest available nmslib version
             */
            V165("165"){
                @Override
                public String indexLibraryVersion() {
                    return KNNConstants.JNI_LIBRARY_NAME;
                }
            };

            static final Version LATEST = V165;

            String buildVersion;

            Version(String buildVersion) {
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
