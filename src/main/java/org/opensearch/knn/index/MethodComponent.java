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

import org.opensearch.common.ValidationException;

import java.util.HashMap;
import java.util.Map;

/**
 * MethodComponent defines the structure of an individual component that can make up an index
 */
public class MethodComponent {

    private String name;
    private Map<String, Parameter<?>> parameters;

    /**
     * Constructor
     *
     * @param builder to build method component
     */
    private MethodComponent(Builder builder) {
        this.name = builder.name;
        this.parameters = builder.parameters;
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
     * Builder class for MethodComponent
     */
    public static class Builder {

        private String name;
        private Map<String, Parameter<?>> parameters;

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
         * Build MethodComponent
         *
         * @return Method Component built from builder
         */
        public MethodComponent build() {
            return new MethodComponent(this);
        }
    }
}
