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
import org.opensearch.knn.training.VectorSpaceInfo;

import java.util.Map;
import java.util.function.BiFunction;
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
    protected BiFunction<T, VectorSpaceInfo, Boolean> validatorWithData;

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
        this.validatorWithData = null;
    }

    public Parameter(String name, T defaultValue, Predicate<T> validator, BiFunction<T, VectorSpaceInfo, Boolean> validatorWithData) {
        this.name = name;
        this.defaultValue = defaultValue;
        this.validator = validator;
        this.validatorWithData = validatorWithData;
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
     * Check if the value passed in is valid, using additional data not present in the value
     *
     * @param value to be checked
     * @param vectorSpaceInfo additional data not present in the value
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public abstract ValidationException validateWithData(Object value, VectorSpaceInfo vectorSpaceInfo);

    /**
     * Boolean method parameter
     */
    public static class BooleanParameter extends Parameter<Boolean> {
        public BooleanParameter(String name, Boolean defaultValue, Predicate<Boolean> validator) {
            super(name, defaultValue, validator);
        }

        public BooleanParameter(
            String name,
            Boolean defaultValue,
            Predicate<Boolean> validator,
            BiFunction<Boolean, VectorSpaceInfo, Boolean> validatorWithData
        ) {
            super(name, defaultValue, validator, validatorWithData);
        }

        @Override
        public ValidationException validate(Object value) {
            ValidationException validationException = null;
            if (!(value instanceof Boolean)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of Boolean for Boolean parameter [%s].", getName())
                );
                return validationException;
            }

            if (!validator.test((Boolean) value)) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("parameter validation failed for Boolean parameter [%s].", getName()));
            }
            return validationException;
        }

        @Override
        public ValidationException validateWithData(Object value, VectorSpaceInfo vectorSpaceInfo) {
            ValidationException validationException = null;
            if (!(value instanceof Boolean)) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("value not of type Boolean for Boolean parameter [%s].", getName()));
                return validationException;
            }

            if (validatorWithData == null) {
                return null;
            }

            if (!validatorWithData.apply((Boolean) value, vectorSpaceInfo)) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("parameter validation failed for Boolean parameter [%s].", getName()));
            }

            return validationException;
        }
    }

    /**
     * Integer method parameter
     */
    public static class IntegerParameter extends Parameter<Integer> {
        public IntegerParameter(String name, Integer defaultValue, Predicate<Integer> validator) {
            super(name, defaultValue, validator);
        }

        public IntegerParameter(
            String name,
            Integer defaultValue,
            Predicate<Integer> validator,
            BiFunction<Integer, VectorSpaceInfo, Boolean> validatorWithData
        ) {
            super(name, defaultValue, validator, validatorWithData);
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

        @Override
        public ValidationException validateWithData(Object value, VectorSpaceInfo vectorSpaceInfo) {
            ValidationException validationException = null;
            if (!(value instanceof Integer)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of Integer for Integer parameter [%s].", getName())
                );
                return validationException;
            }

            if (validatorWithData == null) {
                return null;
            }

            if (!validatorWithData.apply((Integer) value, vectorSpaceInfo)) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("parameter validation failed for Integer parameter [%s].", getName()));
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

        public StringParameter(
            String name,
            String defaultValue,
            Predicate<String> validator,
            BiFunction<String, VectorSpaceInfo, Boolean> validatorWithData
        ) {
            super(name, defaultValue, validator, validatorWithData);
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

        @Override
        public ValidationException validateWithData(Object value, VectorSpaceInfo vectorSpaceInfo) {
            ValidationException validationException = null;
            if (!(value instanceof String)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of String for String parameter [%s].", getName())
                );
                return validationException;
            }

            if (validatorWithData == null) {
                return null;
            }

            if (!validatorWithData.apply((String) value, vectorSpaceInfo)) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("parameter validation failed for String parameter [%s].", getName()));
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
            }, (methodComponentContext, vectorSpaceInfo) -> {
                if (!methodComponents.containsKey(methodComponentContext.getName())) {
                    return false;
                }
                return methodComponents.get(methodComponentContext.getName())
                    .validateWithData(methodComponentContext, vectorSpaceInfo) == null;
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

        @Override
        public ValidationException validateWithData(Object value, VectorSpaceInfo vectorSpaceInfo) {
            ValidationException validationException = null;
            if (!(value instanceof MethodComponentContext)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of for MethodComponentContext parameter [%s].", getName())
                );
                return validationException;
            }

            if (validatorWithData == null) {
                return null;
            }

            if (!validatorWithData.apply((MethodComponentContext) value, vectorSpaceInfo)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("parameter validation failed for MethodComponentContext parameter [%s].", getName())
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
