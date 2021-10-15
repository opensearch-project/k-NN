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

package org.opensearch.knn.index;

import org.opensearch.common.TriFunction;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * MethodComponent defines the structure of an individual component that can make up an index
 */
public class MethodComponent {

    private String name;
    private Map<String, Parameter<?>> parameters;
    private BiFunction<MethodComponent, MethodComponentContext, Map<String, Object>> mapGenerator;
    private TriFunction<MethodComponent, MethodComponentContext, Integer, Long> overheadInKBEstimator;
    final private boolean requiresTraining;

    /**
     * Constructor
     *
     * @param builder to build method component
     */
    private MethodComponent(Builder builder) {
        this.name = builder.name;
        this.parameters = builder.parameters;
        this.mapGenerator = builder.mapGenerator;
        this.overheadInKBEstimator = builder.overheadInKBEstimator;
        this.requiresTraining = builder.requiresTraining;
    }

    /**
     * Get the name of the component
     *
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * Get the parameters for the component
     *
     * @return parameters
     */
    public Map<String, Parameter<?>> getParameters() {
        return parameters;
    }


    /**
     * Parse methodComponentContext into a map that the library can use to configure the method
     *
     * @param methodComponentContext from which to generate map
     * @return Method component as a map
     */
    public Map<String, Object> getAsMap(MethodComponentContext methodComponentContext) {
        if (mapGenerator == null) {
            Map<String, Object> parameterMap = new HashMap<>();
            parameterMap.put(KNNConstants.NAME, methodComponentContext.getName());
            parameterMap.put(KNNConstants.PARAMETERS, methodComponentContext.getParameters());
            return parameterMap;
        }
        return mapGenerator.apply(this, methodComponentContext);
    }

    /**
     * Validate that the methodComponentContext is a valid configuration for this methodComponent
     *
     * @param methodComponentContext to be validated
     */
    public void validate(MethodComponentContext methodComponentContext) {
        Map<String, Object> providedParameters = methodComponentContext.getParameters();

        if (providedParameters == null) {
            return;
        }

        for (Map.Entry<String, Object> parameter : providedParameters.entrySet()) {
            if (!parameters.containsKey(parameter.getKey())) {
                throw new ValidationException();
            }

            parameters.get(parameter.getKey()).validate(parameter.getValue());
        }
    }
    /**
     * gets requiresTraining value
     *
     * @return requiresTraining
     */
    public boolean isTrainingRequired(MethodComponentContext methodComponentContext) {
        if (requiresTraining) {
            return true;
        }

        // Check if any of the parameters the user provided require training. For example, PQ as an encoder.
        // If so, return true as well
        Map<String, Object> providedParameters = methodComponentContext.getParameters();
        if (providedParameters == null) {
            return false;
        }

        for (Map.Entry<String, Object> providedParameter : providedParameters.entrySet()) {
            // MethodComponentContextParameters are parameters that are MethodComponentContexts.
            // MethodComponent may or may not require training. So, we have to check if the parameter requires training.
            // If the parameter does not exist, the parameter estimate will be skipped. It is not this function's job
            // to validate the parameters.
            Parameter<?> parameter = parameters.get(providedParameter.getKey());
            if (!(parameter instanceof Parameter.MethodComponentContextParameter)) {
                continue;
            }

            Parameter.MethodComponentContextParameter methodParameter = (Parameter.MethodComponentContextParameter) parameter;
            Object providedValue = providedParameter.getValue();
            if (!(providedValue instanceof MethodComponentContext)) {
                continue;
            }

            MethodComponentContext parameterMethodComponentContext = (MethodComponentContext) providedValue;
            MethodComponent methodComponent = methodParameter.getMethodComponent(parameterMethodComponentContext.getName());
            if (methodComponent.isTrainingRequired(parameterMethodComponentContext)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Estimates the overhead in KB
     *
     * @param methodComponentContext context to make estimate for
     * @param dimension dimension to make estimate with
     * @return overhead estimate in kb
     */
    public int estimateOverheadInKB(MethodComponentContext methodComponentContext, int dimension) {
        // Assume we have the following KNNMethodContext:
        // "method": {
        //      "name":"METHOD_1",
        //      "engine":"faiss",
        //      "space_type": "l2",
        //      "parameters":{
        //         "P1":1,
        //         "P2":{
        //              "name":"METHOD_2",
        //              "parameters":{
        //                 "P3":2
        //              }
        //         }
        //     }
        // }
        //
        // First, we get the overhead estimate of METHOD_1. Then, we add the overhead
        // estimate for METHOD_2 by looping over parameters of METHOD_1.
        
        long size = overheadInKBEstimator.apply(this, methodComponentContext, dimension);

        // Check if any of the parameters add overhead
        Map<String, Object> providedParameters = methodComponentContext.getParameters();
        if (providedParameters == null) {
            return Math.toIntExact(size);
        }

        for (Map.Entry<String, Object> providedParameter : providedParameters.entrySet()) {
            // MethodComponentContextParameters are parameters that are MethodComponentContexts. We need to check if
            // these parameters add overhead. If the parameter does not exist, the parameter estimate will be skipped.
            // It is not this function's job to validate the parameters.
            Parameter<?> parameter = parameters.get(providedParameter.getKey());
            if (!(parameter instanceof Parameter.MethodComponentContextParameter)) {
                continue;
            }

            Parameter.MethodComponentContextParameter methodParameter = (Parameter.MethodComponentContextParameter) parameter;
            Object providedValue = providedParameter.getValue();
            if (!(providedValue instanceof MethodComponentContext)) {
                continue;
            }

            MethodComponentContext parameterMethodComponentContext = (MethodComponentContext) providedValue;
            MethodComponent methodComponent = methodParameter.getMethodComponent(parameterMethodComponentContext.getName());
            size += methodComponent.estimateOverheadInKB(parameterMethodComponentContext, dimension);
        }

        return Math.toIntExact(size);
    }

    /**
     * Builder class for MethodComponent
     */
    public static class Builder {

        private String name;
        private Map<String, Parameter<?>> parameters;
        private BiFunction<MethodComponent, MethodComponentContext, Map<String, Object>> mapGenerator;
        private TriFunction<MethodComponent, MethodComponentContext, Integer, Long> overheadInKBEstimator;
        private boolean requiresTraining;

        /**
         * Method to get a Builder instance
         *
         * @param name of method component
         * @return Builder instance
         */
        public static Builder builder(String name) {
            return new MethodComponent.Builder(name);
        }

        private Builder(String name) {
            this.name = name;
            this.parameters = new HashMap<>();
            this.mapGenerator = null;
            this.overheadInKBEstimator = (mc, mcc, d) -> 0L;
        }

        /**
         * Add parameter entry to parameters map
         *
         * @param parameterName name of the parameter
         * @param parameter parameter to be added
         * @return this builder
         */
        public Builder addParameter(String parameterName, Parameter<?> parameter) {
            this.parameters.put(parameterName, parameter);
            return this;
        }

        /**
         * Set the function used to parse a MethodComponentContext as a map
         *
         * @param mapGenerator function to parse a MethodComponentContext as a map
         * @return this builder
         */
        public Builder setMapGenerator(BiFunction<MethodComponent, MethodComponentContext, Map<String, Object>> mapGenerator) {
            this.mapGenerator = mapGenerator;
            return this;
        }

        /**
         * set requiresTraining
         * @param requiresTraining parameter to be set
         * @return Builder instance
         */
        public Builder setRequiresTraining(boolean requiresTraining){
            this.requiresTraining = requiresTraining;
            return this;
        }

        /**
         * Set the function used to compute an estimate of the size of the component in KB
         *
         * @param overheadInKBEstimator function that will compute the estimation
         * @return Builder instance
         */
        public Builder setOverheadInKBEstimator(TriFunction<MethodComponent, MethodComponentContext, Integer, Long> overheadInKBEstimator) {
            this.overheadInKBEstimator = overheadInKBEstimator;
            return this;
        }

        /**
         * Build MethodComponent
         *
         * @return Method Component built from builder
         */
        public MethodComponent build() {
            return new MethodComponent(this);
        }
    }
}
