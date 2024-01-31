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

import java.util.Map;
import java.util.function.Predicate;

/**
 * Parameter that can be set for a method component
 *
 * @param <T> Type parameter takes
 */
public abstract class Parameter<T> {

    private String name;
    private T defaultValue;
    protected Predicate<T> validator;

    /**
     * Constructor
     *
     * @param name of the parameter
     * @param defaultValue of the parameter
     * @param validator used to validate a parameter value passed
     */
    public Parameter(String name, T defaultValue, Predicate<T> validator) {
        this.name = name;
        this.defaultValue = defaultValue;
        this.validator = validator;
    }

    /**
     * Getter for parameter name
     *
     * @return parameter name
     */
    public String getName() {
        return name;
    }

    /**
     * Get default value for parameter
     *
     * @return default value of the parameter
     */
    public T getDefaultValue() {
        return defaultValue;
    }

    /**
     * Check if the value passed in is valid
     *
     * @param value to be checked
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public abstract ValidationException validate(Object value);

    /**
     * Integer method parameter
     */
    public static class IntegerParameter extends Parameter<Integer> {
        public IntegerParameter(String name, Integer defaultValue, Predicate<Integer> validator) {
            super(name, defaultValue, validator);
        }

        @Override
        public ValidationException validate(Object value) {
            ValidationException validationException = null;
            if (!(value instanceof Integer)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("Value not of type Integer for Integer " + "parameter \"%s\".", getName())
                );
                return validationException;
            }

            if (!validator.test((Integer) value)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("Parameter validation failed for Integer " + "parameter \"%s\".", getName())
                );
            }
            return validationException;
        }
    }

    /**
     * String method parameter
     */
    public static class StringParameter extends Parameter<String> {

        /**
         * Constructor
         *
         * @param name         of the parameter
         * @param defaultValue value to assign if the parameter is not set
         * @param validator    used to validate the parameter value passed
         */
        public StringParameter(String name, String defaultValue, Predicate<String> validator) {
            super(name, defaultValue, validator);
        }

        /**
         * Check if the value passed in is valid
         *
         * @param value to be checked
         * @return ValidationException produced by validation errors; null if no validations errors.
         */
        @Override
        public ValidationException validate(Object value) {
            ValidationException validationException = null;
            if (!(value instanceof String)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("Value not of type String for String " + "parameter \"%s\".", getName())
                );
                return validationException;
            }

            if (!validator.test((String) value)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("Parameter validation failed for String " + "parameter \"%s\".", getName())
                );
            }
            return validationException;
        }
    }

    /**
     * MethodContext parameter. Some methods require sub-methods in order to implement some kind of functionality. For
     *  instance, faiss methods can contain an encoder along side the approximate nearest neighbor function to compress
     *  the input. This parameter makes it possible to add sub-methods to methods to support this kind of functionality
     */
    public static class MethodComponentContextParameter extends Parameter<MethodComponentContext> {

        private Map<String, MethodComponent> methodComponents;

        /**
         * Constructor
         *
         * @param name of the parameter
         * @param defaultValue value to assign this parameter if it is not set
         * @param methodComponents valid components that the MethodComponentContext can map to
         */
        public MethodComponentContextParameter(
            String name,
            MethodComponentContext defaultValue,
            Map<String, MethodComponent> methodComponents
        ) {
            super(name, defaultValue, methodComponentContext -> {
                if (!methodComponents.containsKey(methodComponentContext.getName())) {
                    return false;
                }

                return methodComponents.get(methodComponentContext.getName()).validate(methodComponentContext) == null;
            });
            this.methodComponents = methodComponents;
        }

        @Override
        public ValidationException validate(Object value) {
            ValidationException validationException = null;
            if (!(value instanceof MethodComponentContext)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("Value not of type MethodComponentContext for" + " MethodComponentContext parameter \"%s\".", getName())
                );
                return validationException;
            }

            if (!validator.test((MethodComponentContext) value)) {
                validationException = new ValidationException();
                validationException.addValidationError("Parameter validation failed.");
                validationException.addValidationError(
                    String.format("Parameter validation failed for " + "MethodComponentContext parameter \"%s\".", getName())
                );
            }

            return validationException;
        }

        /**
         * Get method component by name
         *
         * @param name name of method component
         * @return MethodComponent that name maps to
         */
        public MethodComponent getMethodComponent(String name) {
            return methodComponents.get(name);
        }
    }
}
