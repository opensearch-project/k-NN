/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import lombok.AllArgsConstructor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.SpaceType;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_IVF_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NSG_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_NSG;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_LIMIT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_LIMIT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_R;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * Implements NativeLibrary for the faiss native library
 */
class Faiss extends NativeLibrary {

    private final static String CURRENT_VERSION = "165";

    // Map that overrides OpenSearch score translation by space type of scores returned by faiss
    private final static Map<SpaceType, Function<Float, Float>> SCORE_TRANSLATIONS = ImmutableMap.of(
        SpaceType.INNER_PRODUCT,
        rawScore -> SpaceType.INNER_PRODUCT.scoreTranslation(-1 * rawScore)
    );

    // Define encoders supported by faiss
    private final static MethodComponentContext ENCODER_DEFAULT = new MethodComponentContext(
        KNNConstants.ENCODER_FLAT,
        Collections.emptyMap()
    );

    // TODO: To think about in future: for PQ, if dimension is not divisible by code count, PQ will fail. Right now,
    // we do not have a way to base validation off of dimension. Failure will happen during training in JNI.
    private final static Map<String, MethodComponent> encoderComponents = ImmutableMap.of(
        KNNConstants.ENCODER_FLAT,
        MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
            .setMapGenerator(
                ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                    KNNConstants.FAISS_FLAT_DESCRIPTION,
                    methodComponent,
                    methodComponentContext
                ).build())
            )
            .build(),
        KNNConstants.ENCODER_PQ,
        MethodComponent.Builder.builder(KNNConstants.ENCODER_PQ)
            .addParameter(
                ENCODER_PARAMETER_PQ_M,
                new Parameter.IntegerParameter(
                    ENCODER_PARAMETER_PQ_M,
                    ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT,
                    v -> v > 0 && v < ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT
                )
            )
            .addParameter(
                ENCODER_PARAMETER_PQ_CODE_SIZE,
                new Parameter.IntegerParameter(
                    ENCODER_PARAMETER_PQ_CODE_SIZE,
                    ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT,
                    v -> v > 0 && v < ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT
                )
            )
            .setRequiresTraining(true)
            .setMapGenerator(
                ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                    FAISS_PQ_DESCRIPTION,
                    methodComponent,
                    methodComponentContext
                ).addParameter(ENCODER_PARAMETER_PQ_M, "", "").addParameter(ENCODER_PARAMETER_PQ_CODE_SIZE, "x", "").build())
            )
            .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                // Size estimate formula: (4 * d * 2^code_size) / 1024 + 1

                // Get value of code size passed in by user
                Object codeSizeObject = methodComponentContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);

                // If not specified, get default value of code size
                if (codeSizeObject == null) {
                    Parameter<?> codeSizeParameter = methodComponent.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);
                    if (codeSizeParameter == null) {
                        throw new IllegalStateException(
                            String.format("%s  is not a valid parameter. This is a bug.", ENCODER_PARAMETER_PQ_CODE_SIZE)
                        );
                    }

                    codeSizeObject = codeSizeParameter.getDefaultValue();
                }

                if (!(codeSizeObject instanceof Integer)) {
                    throw new IllegalStateException(String.format("%s must be an integer.", ENCODER_PARAMETER_PQ_CODE_SIZE));
                }

                int codeSize = (Integer) codeSizeObject;
                return ((4L * (1L << codeSize) * dimension) / BYTES_PER_KILOBYTES) + 1;
            })
            .build()
    );

    // Define methods supported by faiss
    private final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
        METHOD_HNSW,
        KNNMethod.Builder.builder(
            MethodComponent.Builder.builder(METHOD_HNSW)
                .addParameter(
                    METHOD_PARAMETER_M,
                    new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0)
                )
                .addParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    new Parameter.IntegerParameter(
                        METHOD_PARAMETER_EF_CONSTRUCTION,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                        v -> v > 0
                    )
                )
                .addParameter(
                    METHOD_PARAMETER_EF_SEARCH,
                    new Parameter.IntegerParameter(
                        METHOD_PARAMETER_EF_SEARCH,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                        v -> v > 0
                    )
                )
                .addParameter(
                    METHOD_ENCODER_PARAMETER,
                    new Parameter.MethodComponentContextParameter(METHOD_ENCODER_PARAMETER, ENCODER_DEFAULT, encoderComponents)
                )
                .setMapGenerator(
                    ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                        FAISS_HNSW_DESCRIPTION,
                        methodComponent,
                        methodComponentContext
                    ).addParameter(METHOD_PARAMETER_M, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "").build())
                )
                .build()
        ).addSpaces(SpaceType.L2, SpaceType.INNER_PRODUCT).build(),
        METHOD_IVF,
        KNNMethod.Builder.builder(
            MethodComponent.Builder.builder(METHOD_IVF)
                .addParameter(
                    METHOD_PARAMETER_NPROBES,
                    new Parameter.IntegerParameter(
                        METHOD_PARAMETER_NPROBES,
                        METHOD_PARAMETER_NPROBES_DEFAULT,
                        v -> v > 0 && v < METHOD_PARAMETER_NPROBES_LIMIT
                    )
                )
                .addParameter(
                    METHOD_PARAMETER_NLIST,
                    new Parameter.IntegerParameter(
                        METHOD_PARAMETER_NLIST,
                        METHOD_PARAMETER_NLIST_DEFAULT,
                        v -> v > 0 && v < METHOD_PARAMETER_NLIST_LIMIT
                    )
                )
                .addParameter(
                    METHOD_ENCODER_PARAMETER,
                    new Parameter.MethodComponentContextParameter(METHOD_ENCODER_PARAMETER, ENCODER_DEFAULT, encoderComponents)
                )
                .setRequiresTraining(true)
                .setMapGenerator(
                    ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                        FAISS_IVF_DESCRIPTION,
                        methodComponent,
                        methodComponentContext
                    ).addParameter(METHOD_PARAMETER_NLIST, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "").build())
                )
                .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                    // Size estimate formula: (4 * nlists * d) / 1024 + 1

                    // Get value of nlists passed in by user
                    Object nlistObject = methodComponentContext.getParameters().get(METHOD_PARAMETER_NLIST);

                    // If not specified, get default value of nlist
                    if (nlistObject == null) {
                        Parameter<?> nlistParameter = methodComponent.getParameters().get(METHOD_PARAMETER_NLIST);
                        if (nlistParameter == null) {
                            throw new IllegalStateException(
                                String.format("%s  is not a valid parameter. This is a bug.", METHOD_PARAMETER_NLIST)
                            );
                        }

                        nlistObject = nlistParameter.getDefaultValue();
                    }

                    if (!(nlistObject instanceof Integer)) {
                        throw new IllegalStateException(String.format("%s must be an integer.", METHOD_PARAMETER_NLIST));
                    }

                    int centroids = (Integer) nlistObject;
                    return ((4L * centroids * dimension) / BYTES_PER_KILOBYTES) + 1;
                })
                .build()
        ).addSpaces(SpaceType.L2, SpaceType.INNER_PRODUCT).build(),
        METHOD_NSG,
        KNNMethod.Builder.builder(
            MethodComponent.Builder.builder(METHOD_NSG)
                .addParameter(
                    METHOD_PARAMETER_R,
                    new Parameter.IntegerParameter(METHOD_PARAMETER_R, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_R, v -> v > 0)
                )
                .addParameter(
                    METHOD_ENCODER_PARAMETER,
                    new Parameter.MethodComponentContextParameter(METHOD_ENCODER_PARAMETER, ENCODER_DEFAULT, encoderComponents)
                )
                .setMapGenerator(
                    ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                        FAISS_NSG_DESCRIPTION,
                        methodComponent,
                        methodComponentContext
                    ).addParameter(METHOD_PARAMETER_R, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "").build())
                )
                .build()
        ).addSpaces(SpaceType.L2, SpaceType.INNER_PRODUCT).build()

    );

    final static Faiss INSTANCE = new Faiss(METHODS, SCORE_TRANSLATIONS, CURRENT_VERSION, KNNConstants.FAISS_EXTENSION);

    /**
     * Constructor for Faiss
     *
     * @param methods                   map of methods the native library supports
     * @param scoreTranslation          Map of translation of space type to scores returned by the library
     * @param currentVersion            String representation of current version of the library
     * @param extension                 String representing the extension that library files should use
     */
    private Faiss(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String currentVersion,
        String extension
    ) {
        super(methods, scoreTranslation, currentVersion, extension);
    }

    /**
     * MethodAsMap builder is used to create the map that will be passed to the jni to create the faiss index.
     * Faiss's index factory takes an "index description" that it uses to build the index. In this description,
     * some parameters of the index can be configured; others need to be manually set. MethodMap builder creates
     * the index description from a set of parameters and removes them from the map. On build, it sets the index
     * description in the map and returns the processed map
     */
    @AllArgsConstructor
    static class MethodAsMapBuilder {
        String indexDescription;
        MethodComponent methodComponent;
        Map<String, Object> methodAsMap;

        /**
         * Add a parameter that will be used in the index description for the given method component
         *
         * @param parameterName name of the parameter
         * @param prefix to append to the index description before the parameter
         * @param suffix to append to the index description after the parameter
         * @return this builder
         */
        @SuppressWarnings("unchecked")
        MethodAsMapBuilder addParameter(String parameterName, String prefix, String suffix) {
            indexDescription += prefix;

            // When we add a parameter, what we are doing is taking it from the methods parameter and building it
            // into the index description string faiss uses to create the index.
            Map<String, Object> methodParameters = (Map<String, Object>) methodAsMap.get(PARAMETERS);
            Parameter<?> parameter = methodComponent.getParameters().get(parameterName);
            Object value = methodParameters.containsKey(parameterName) ? methodParameters.get(parameterName) : parameter.getDefaultValue();

            // Recursion is needed if the parameter is a method component context itself.
            if (parameter instanceof Parameter.MethodComponentContextParameter) {
                MethodComponentContext subMethodComponentContext = (MethodComponentContext) value;
                MethodComponent subMethodComponent = ((Parameter.MethodComponentContextParameter) parameter).getMethodComponent(
                    subMethodComponentContext.getName()
                );

                Map<String, Object> subMethodAsMap = subMethodComponent.getAsMap(subMethodComponentContext);
                indexDescription += subMethodAsMap.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER);
                subMethodAsMap.remove(KNNConstants.INDEX_DESCRIPTION_PARAMETER);

                // We replace parameterName with the map that contains only parameters that are not included in
                // the method description
                methodParameters.put(parameterName, subMethodAsMap);
            } else {
                // Just add the value to the method description and remove from map
                indexDescription += value;
                methodParameters.remove(parameterName);
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

        static MethodAsMapBuilder builder(
            String baseDescription,
            MethodComponent methodComponent,
            MethodComponentContext methodComponentContext
        ) {
            Map<String, Object> initialMap = new HashMap<>();
            initialMap.put(NAME, methodComponent.getName());
            initialMap.put(PARAMETERS, MethodComponent.getParameterMapWithDefaultsAdded(methodComponentContext, methodComponent));
            return new MethodAsMapBuilder(baseDescription, methodComponent, initialMap);
        }
    }
}
