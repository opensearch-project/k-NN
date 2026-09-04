/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import lombok.AllArgsConstructor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * Builds the faiss index-factory description (e.g. {@code "SVSVamana64,LVQ4x4"}) and parameter map. Copy of
 * core's package-private {@code MethodAsMapBuilder} plus {@link #dropTrailingDescriptionToken(String)}.
 */
@AllArgsConstructor
class SvsMethodAsMapBuilder {
    String indexDescription;
    MethodComponent methodComponent;
    Map<String, Object> methodAsMap;
    KNNMethodConfigContext knnMethodConfigContext;
    QuantizationConfig quantizationConfig;

    @SuppressWarnings("unchecked")
    SvsMethodAsMapBuilder addParameter(String parameterName, String prefix, String suffix) {
        indexDescription += prefix;

        Map<String, Object> methodParameters = (Map<String, Object>) methodAsMap.get(PARAMETERS);
        Parameter<?> parameter = methodComponent.getParameters().get(parameterName);
        Object value = methodParameters.containsKey(parameterName) ? methodParameters.get(parameterName) : parameter.getDefaultValue();

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
            indexDescription += value;
        }

        indexDescription += suffix;
        return this;
    }

    /**
     * Drops a trailing {@code ",<token>"} from the index description if present. SVS methods still call
     * {@link #addParameter} for the default {@code flat} encoder so it is normalized into the method map and
     * serializes, then use this to strip the {@code ,Flat} suffix that the native factory does not accept.
     */
    SvsMethodAsMapBuilder dropTrailingDescriptionToken(String token) {
        String suffix = "," + token;
        if (indexDescription.endsWith(suffix)) {
            indexDescription = indexDescription.substring(0, indexDescription.length() - suffix.length());
        }
        return this;
    }

    KNNLibraryIndexingContext build() {
        methodAsMap.put(KNNConstants.INDEX_DESCRIPTION_PARAMETER, indexDescription);
        return KNNLibraryIndexingContextImpl.builder().parameters(methodAsMap).quantizationConfig(quantizationConfig).build();
    }

    static SvsMethodAsMapBuilder builder(
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
        return new SvsMethodAsMapBuilder(baseDescription, methodComponent, initialMap, knnMethodConfigContext, QuantizationConfig.EMPTY);
    }
}
