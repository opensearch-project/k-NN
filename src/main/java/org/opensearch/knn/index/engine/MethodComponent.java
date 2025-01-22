/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Getter;
import org.opensearch.Version;
import org.opensearch.common.TriFunction;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.index.engine.validation.ParameterValidator.validateParameters;

/**
 * MethodComponent defines the structure of an individual component that can make up an index
 */
public class MethodComponent {

    @Getter
    private final String name;
    @Getter
    private final Map<String, Parameter<?>> parameters;
    private final TriFunction<
        MethodComponent,
        MethodComponentContext,
        KNNMethodConfigContext,
        KNNLibraryIndexingContext> knnLibraryIndexingContextGenerator;
    private final TriFunction<MethodComponent, MethodComponentContext, Integer, Long> overheadInKBEstimator;
    private final boolean requiresTraining;
    private final Set<VectorDataType> supportedVectorDataTypes;

    /**
     * Constructor
     *
     * @param builder to build method component
     */
    private MethodComponent(Builder builder) {
        this.name = builder.name;
        this.parameters = builder.parameters;
        this.knnLibraryIndexingContextGenerator = builder.knnLibraryIndexingContextGenerator;
        this.overheadInKBEstimator = builder.overheadInKBEstimator;
        this.requiresTraining = builder.requiresTraining;
        this.supportedVectorDataTypes = builder.supportedDataTypes;
    }

    /**
     * Parse methodComponentContext into a map that the library can use to configure the method
     *
     * @param methodComponentContext from which to generate map
     * @return Method component as a map
     */
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (knnLibraryIndexingContextGenerator == null) {
            Map<String, Object> parameterMap = new HashMap<>();
            parameterMap.put(KNNConstants.NAME, methodComponentContext.getName());
            parameterMap.put(
                KNNConstants.PARAMETERS,
                getParameterMapWithDefaultsAdded(methodComponentContext, this, knnMethodConfigContext)
            );
            return KNNLibraryIndexingContextImpl.builder().parameters(parameterMap).build();
        }
        return knnLibraryIndexingContextGenerator.apply(this, methodComponentContext, knnMethodConfigContext);
    }

    /**
     * Validate that the methodComponentContext is a valid configuration for this methodComponent
     *
     * @param methodComponentContext to be validated
     * @param knnMethodConfigContext context for the method configuration
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validate(MethodComponentContext methodComponentContext, KNNMethodConfigContext knnMethodConfigContext) {
        Map<String, Object> providedParameters = methodComponentContext.getParameters();

        ValidationException validationException = null;
        if (!supportedVectorDataTypes.contains(knnMethodConfigContext.getVectorDataType())) {
            validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Method \"%s\" is not supported for vector data type \"%s\".",
                    name,
                    knnMethodConfigContext.getVectorDataType()
                )
            );
        }

        ValidationException methodValidationException = validateParameters(parameters, providedParameters, knnMethodConfigContext);

        if (methodValidationException != null) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationErrors(methodValidationException.validationErrors());
        }

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

        private final String name;
        private final Map<String, Parameter<?>> parameters;
        private TriFunction<
            MethodComponent,
            MethodComponentContext,
            KNNMethodConfigContext,
            KNNLibraryIndexingContext> knnLibraryIndexingContextGenerator;
        private TriFunction<MethodComponent, MethodComponentContext, Integer, Long> overheadInKBEstimator;
        private boolean requiresTraining;
        private final Set<VectorDataType> supportedDataTypes;

        /**
         * Method to get a Builder instance
         *
         * @param name of method component
         * @return Builder instance
         */
        public static Builder builder(String name) {
            return new Builder(name);
        }

        private Builder(String name) {
            this.name = name;
            this.parameters = new HashMap<>();
            this.overheadInKBEstimator = (mc, mcc, d) -> 0L;
            this.supportedDataTypes = new HashSet<>();
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
         * @param knnLibraryIndexingContextGenerator function to parse a MethodComponentContext as a knnLibraryIndexingContext
         * @return this builder
         */
        public Builder setKnnLibraryIndexingContextGenerator(
            TriFunction<
                MethodComponent,
                MethodComponentContext,
                KNNMethodConfigContext,
                KNNLibraryIndexingContext> knnLibraryIndexingContextGenerator
        ) {
            this.knnLibraryIndexingContextGenerator = knnLibraryIndexingContextGenerator;
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
         * Adds supported data types to the method component
         *
         * @param dataTypeSet supported data types
         * @return Builder instance
         */
        public Builder addSupportedDataTypes(Set<VectorDataType> dataTypeSet) {
            supportedDataTypes.addAll(dataTypeSet);
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
        MethodComponent methodComponent,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        Map<String, Object> parametersWithDefaultsMap = new HashMap<>();
        Map<String, Object> userProvidedParametersMap = methodComponentContext.getParameters();
        Version indexCreationVersion = knnMethodConfigContext.getVersionCreated();
        Mode mode = knnMethodConfigContext.getMode();
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();

        // Check if the mode is ON_DISK and the compression level is one of the binary quantization levels (x32, x16, or x8).
        // This determines whether to use binary quantization-specific values for parameters like ef_search and ef_construction.
        boolean isOnDiskWithBinaryQuantization = (compressionLevel == CompressionLevel.x32
            || compressionLevel == CompressionLevel.x16
            || compressionLevel == CompressionLevel.x8);

        // Check if the mode is ON_DISK and the compression level is x4 and index created version is >= 2.19.0
        // This determines whether to use faiss byte quantization-specific values for parameters like ef_search and ef_construction.
        boolean isFaissOnDiskWithByteQuantization = compressionLevel == CompressionLevel.x4
            && indexCreationVersion.onOrAfter(Version.V_2_19_0);

        for (Parameter<?> parameter : methodComponent.getParameters().values()) {
            if (methodComponentContext.getParameters().containsKey(parameter.getName())) {
                parametersWithDefaultsMap.put(parameter.getName(), userProvidedParametersMap.get(parameter.getName()));
            } else {
                // Picking the right values for the parameters whose values are different based on different index
                // created version.
                if (parameter.getName().equals(KNNConstants.METHOD_PARAMETER_EF_SEARCH)) {
                    if (isOnDiskWithBinaryQuantization || isFaissOnDiskWithByteQuantization) {
                        parametersWithDefaultsMap.put(parameter.getName(), IndexHyperParametersUtil.getBinaryQuantizationEFSearchValue());
                    } else {
                        parametersWithDefaultsMap.put(
                            parameter.getName(),
                            IndexHyperParametersUtil.getHNSWEFSearchValue(indexCreationVersion)
                        );
                    }
                } else if (parameter.getName().equals(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
                    if (isOnDiskWithBinaryQuantization || isFaissOnDiskWithByteQuantization) {
                        parametersWithDefaultsMap.put(
                            parameter.getName(),
                            IndexHyperParametersUtil.getBinaryQuantizationEFConstructionValue()
                        );
                    } else {
                        parametersWithDefaultsMap.put(
                            parameter.getName(),
                            IndexHyperParametersUtil.getHNSWEFConstructionValue(indexCreationVersion)
                        );
                    }

                } else {
                    Object value = parameter.getDefaultValue();
                    if (value != null) {
                        parametersWithDefaultsMap.put(parameter.getName(), value);
                    }
                }

            }
        }

        return parametersWithDefaultsMap;
    }
}
