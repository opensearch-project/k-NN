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
 * Builds the faiss-style "index description" string (e.g. {@code "SVSVamana64,LVQ4x4"}) + parameter map that
 * the SVS native factory consumes. SVS indices are faiss-format ({@code IndexSVSVamana} is a {@code faiss::Index}),
 * so this assembles the description exactly as faiss's own {@code MethodAsMapBuilder} does.
 *
 * <p>This is an intentional, self-contained copy of {@code org.opensearch.knn.index.engine.faiss.MethodAsMapBuilder}
 * (which is package-private in core). Duplicating it here keeps the SVS tenant from forcing any visibility or
 * API change onto the core faiss package — the core stays untouched and the tenant carries its own copy, the
 * same trade made for the native JNI helpers. It adds one SVS-only helper,
 * {@link #dropTrailingDescriptionToken(String)}, not present in the core class.
 */
@AllArgsConstructor
class SvsMethodAsMapBuilder {
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
    SvsMethodAsMapBuilder addParameter(String parameterName, String prefix, String suffix) {
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

    /**
     * Build
     *
     * @return Method as a map
     */
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
