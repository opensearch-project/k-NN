/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import lombok.AllArgsConstructor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * MethodAsMap builder is used to create the map that will be passed to the jni to create the faiss index.
 * Faiss's index factory takes an "index description" that it uses to build the index. In this description,
 * some parameters of the index can be configured; others need to be manually set. MethodMap builder creates
 * the index description from a set of parameters and removes them from the map. On build, it sets the index
 * description in the map and returns the processed map
 */
@AllArgsConstructor
class MethodAsMapBuilder {
    String indexDescription;
    MethodComponent methodComponent;
    Map<String, Object> methodAsMap;
    KNNMethodConfigContext knnMethodConfigContext;
    QuantizationConfig quantizationConfig;

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

            KNNLibraryIndexingContext knnLibraryIndexingContext = subMethodComponent.getKNNLibraryIndexingContext(
                subMethodComponentContext,
                knnMethodConfigContext
            );
            Map<String, Object> subMethodAsMap = knnLibraryIndexingContext.getLibraryParameters();
            if (subMethodAsMap != null
                && !subMethodAsMap.isEmpty()
                && subMethodAsMap.containsKey(KNNConstants.INDEX_DESCRIPTION_PARAMETER)) {
                indexDescription += subMethodAsMap.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER);
                subMethodAsMap.remove(KNNConstants.INDEX_DESCRIPTION_PARAMETER);
            }

            if (quantizationConfig == null || quantizationConfig == QuantizationConfig.EMPTY) {
                quantizationConfig = knnLibraryIndexingContext.getQuantizationConfig();
            }

            methodParameters.put(parameterName, subMethodAsMap);
        } else {
            // Add the value to the method description
            indexDescription += value;
        }

        indexDescription += suffix;
        return this;
    }

    /**
     * Build
     *
     * @return Method as a map
     */
    KNNLibraryIndexingContext build() {
        methodAsMap.put(KNNConstants.INDEX_DESCRIPTION_PARAMETER, indexDescription);
        return KNNLibraryIndexingContextImpl.builder().parameters(methodAsMap).quantizationConfig(quantizationConfig).build();
    }

    static MethodAsMapBuilder builder(
        String baseDescription,
        MethodComponent methodComponent,
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        Map<String, Object> initialMap = new HashMap<>();
        initialMap.put(NAME, methodComponent.getName());
        initialMap.put(
            PARAMETERS,
            MethodComponent.getParameterMapWithDefaultsAdded(methodComponentContext, methodComponent, knnMethodConfigContext)
        );

        QuantizationConfig quantizationConfig = QuantizationConfig.EMPTY;
        // TODO: Validate if it impacts Lucene
        if (knnMethodConfigContext.getCompressionLevel() == CompressionLevel.x4) {
            quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.EIGHT_BIT).build();
        }
        return new MethodAsMapBuilder(baseDescription, methodComponent, initialMap, knnMethodConfigContext, quantizationConfig);
    }
}
