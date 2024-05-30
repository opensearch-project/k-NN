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

import lombok.Getter;
import org.opensearch.Version;
import org.opensearch.common.TriFunction;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.knn.training.VectorSpaceInfo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * MethodComponent defines the structure of an individual component that can make up an index
 */
public class MethodComponent {

    @Getter
    private String name;
    @Getter
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
     * Parse methodComponentContext into a map that the library can use to configure the method
     *
     * @param methodComponentContext from which to generate map
     * @return Method component as a map
     */
    public Map<String, Object> getAsMap(MethodComponentContext methodComponentContext) {
        if (mapGenerator == null) {
            Map<String, Object> parameterMap = new HashMap<>();
            parameterMap.put(KNNConstants.NAME, methodComponentContext.getName());
            parameterMap.put(KNNConstants.PARAMETERS, getParameterMapWithDefaultsAdded(methodComponentContext, this));
            return parameterMap;
        }
        return mapGenerator.apply(this, methodComponentContext);
    }

    /**
     * Validate that the methodComponentContext is a valid configuration for this methodComponent
     *
     * @param methodComponentContext to be validated
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validate(MethodComponentContext methodComponentContext) {
        Map<String, Object> providedParameters = methodComponentContext.getParameters();
        List<String> errorMessages = new ArrayList<>();

        if (providedParameters == null) {
            return null;
        }

        ValidationException parameterValidation;
        for (Map.Entry<String, Object> parameter : providedParameters.entrySet()) {
            if (!parameters.containsKey(parameter.getKey())) {
                errorMessages.add(String.format("Invalid parameter for method \"%s\".", getName()));
                continue;
            }

            parameterValidation = parameters.get(parameter.getKey()).validate(parameter.getValue());
            if (parameterValidation != null) {
                errorMessages.addAll(parameterValidation.validationErrors());
            }
        }

        if (errorMessages.isEmpty()) {
            return null;
        }

        ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
    }

    /**
     * Validate that the methodComponentContext is a valid configuration for this methodComponent, using additional data not present in the method component context
     *
     * @param methodComponentContext to be validated
     * @param vectorSpaceInfo additional data not present in the method component context
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validateWithData(MethodComponentContext methodComponentContext, VectorSpaceInfo vectorSpaceInfo) {
        Map<String, Object> providedParameters = methodComponentContext.getParameters();
        List<String> errorMessages = new ArrayList<>();

        if (providedParameters == null) {
            return null;
        }

        ValidationException parameterValidation;
        for (Map.Entry<String, Object> parameter : providedParameters.entrySet()) {
            if (!parameters.containsKey(parameter.getKey())) {
                errorMessages.add(String.format("Invalid parameter for method \"%s\".", getName()));
                continue;
            }

            parameterValidation = parameters.get(parameter.getKey()).validateWithData(parameter.getValue(), vectorSpaceInfo);
            if (parameterValidation != null) {
                errorMessages.addAll(parameterValidation.validationErrors());
            }
        }

        if (errorMessages.isEmpty()) {
            return null;
        }

        ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
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
        if (providedParameters == null || providedParameters.isEmpty()) {
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
        // "name":"METHOD_1",
        // "engine":"faiss",
        // "space_type": "l2",
        // "parameters":{
        // "P1":1,
        // "P2":{
        // "name":"METHOD_2",
        // "parameters":{
        // "P3":2
        // }
        // }
        // }
        // }
        //
        // First, we get the overhead estimate of METHOD_1. Then, we add the overhead
        // estimate for METHOD_2 by looping over parameters of METHOD_1.

        long size = overheadInKBEstimator.apply(this, methodComponentContext, dimension);

        // Check if any of the parameters add overhead
        Map<String, Object> providedParameters = methodComponentContext.getParameters();
        if (providedParameters == null || providedParameters.isEmpty()) {
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
        public Builder setRequiresTraining(boolean requiresTraining) {
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

    /**
     * Returns a map of the user provided parameters in addition to default parameters the user may not have passed
     *
     * @param methodComponentContext context containing user provided parameter
     * @param methodComponent component containing method parameters and defaults
     * @return Map of user provided parameters with defaults filled in as needed
     */
    public static Map<String, Object> getParameterMapWithDefaultsAdded(
        MethodComponentContext methodComponentContext,
        MethodComponent methodComponent
    ) {
        Map<String, Object> parametersWithDefaultsMap = new HashMap<>();
        Map<String, Object> userProvidedParametersMap = methodComponentContext.getParameters();
        Version indexCreationVersion = methodComponentContext.getIndexVersion();
        for (Parameter<?> parameter : methodComponent.getParameters().values()) {
            if (methodComponentContext.getParameters().containsKey(parameter.getName())) {
                parametersWithDefaultsMap.put(parameter.getName(), userProvidedParametersMap.get(parameter.getName()));
            } else {
                // Picking the right values for the parameters whose values are different based on different index
                // created version.
                if (parameter.getName().equals(KNNConstants.METHOD_PARAMETER_EF_SEARCH)) {
                    parametersWithDefaultsMap.put(parameter.getName(), IndexHyperParametersUtil.getHNSWEFSearchValue(indexCreationVersion));
                } else if (parameter.getName().equals(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
                    parametersWithDefaultsMap.put(
                        parameter.getName(),
                        IndexHyperParametersUtil.getHNSWEFConstructionValue(indexCreationVersion)
                    );
                } else {
                    parametersWithDefaultsMap.put(parameter.getName(), parameter.getDefaultValue());
                }

            }
        }

        return parametersWithDefaultsMap;
    }
}
