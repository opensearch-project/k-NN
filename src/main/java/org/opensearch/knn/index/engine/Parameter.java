/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Getter;
import org.opensearch.common.ValidationException;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * Parameter that can be set for a method component
 *
 * @param <T> Type parameter takes
 */
public abstract class Parameter<T> {

    @Getter
    private final String name;
    @Getter
    private final T defaultValue;
    protected BiFunction<T, KNNMethodConfigContext, Boolean> validator;

    /**
     * Constructor
     *
     * @param name of the parameter
     * @param defaultValue of the parameter
     * @param validator used to validate a parameter value passed
     */
    public Parameter(String name, T defaultValue, BiFunction<T, KNNMethodConfigContext, Boolean> validator) {
        this.name = name;
        this.defaultValue = defaultValue;
        this.validator = validator;
    }

    /**
     * Check if the value passed in is valid
     *
     * @param value to be checked
     * @param knnMethodConfigContext context for the validation
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public abstract ValidationException validate(Object value, KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Boolean method parameter
     */
    public static class BooleanParameter extends Parameter<Boolean> {
        public BooleanParameter(String name, Boolean defaultValue, BiFunction<Boolean, KNNMethodConfigContext, Boolean> validator) {
            super(name, defaultValue, validator);
        }

        @Override
        public ValidationException validate(Object value, KNNMethodConfigContext knnMethodConfigContext) {
            ValidationException validationException = null;
            if (!(value instanceof Boolean)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of Boolean for Boolean parameter [%s].", getName())
                );
                return validationException;
            }

            if (!validator.apply((Boolean) value, knnMethodConfigContext)) {
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
        public IntegerParameter(String name, Integer defaultValue, BiFunction<Integer, KNNMethodConfigContext, Boolean> validator) {
            super(name, defaultValue, validator);
        }

        @Override
        public ValidationException validate(Object value, KNNMethodConfigContext knnMethodConfigContext) {
            ValidationException validationException = null;
            if (!(value instanceof Integer)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of Integer for Integer parameter [%s].", getName())
                );
                return validationException;
            }

            if (!validator.apply((Integer) value, knnMethodConfigContext)) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("parameter validation failed for Integer parameter [%s].", getName()));
            }

            return validationException;
        }
    }

    /**
     * Double method parameter
     */
    public static class DoubleParameter extends Parameter<Double> {
        public DoubleParameter(String name, Double defaultValue, BiFunction<Double, KNNMethodConfigContext, Boolean> validator) {
            super(name, defaultValue, validator);
        }

        @Override
        public ValidationException validate(Object value, KNNMethodConfigContext knnMethodConfigContext) {
            if (Objects.isNull(value)) {
                String validationErrorMsg = String.format(Locale.ROOT, "Null value provided for Double " + "parameter \"%s\".", getName());
                return getValidationException(validationErrorMsg);
            }

            if (value.equals(0)) value = 0.0;

            if (!(value instanceof Double)) {
                String validationErrorMsg = String.format(
                    Locale.ROOT,
                    "value is not an instance of Double for Double parameter [%s].",
                    getName()
                );
                return getValidationException(validationErrorMsg);
            }

            if (!validator.apply((Double) value, knnMethodConfigContext)) {
                String validationErrorMsg = String.format(Locale.ROOT, "parameter validation failed for Double parameter [%s].", getName());
                return getValidationException(validationErrorMsg);
            }
            return null;
        }

        private ValidationException getValidationException(String validationErrorMsg) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(validationErrorMsg);
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
        public StringParameter(String name, String defaultValue, BiFunction<String, KNNMethodConfigContext, Boolean> validator) {
            super(name, defaultValue, validator);
        }

        @Override
        public ValidationException validate(Object value, KNNMethodConfigContext knnMethodConfigContext) {
            ValidationException validationException = null;
            if (!(value instanceof String)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of String for String parameter [%s].", getName())
                );
                return validationException;
            }

            if (!validator.apply((String) value, knnMethodConfigContext)) {
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

        private final Map<String, MethodComponent> methodComponents;

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
            super(name, defaultValue, (methodComponentContext, knnMethodConfigContext) -> {
                if (!methodComponents.containsKey(methodComponentContext.getName())) {
                    return false;
                }
                return methodComponents.get(methodComponentContext.getName())
                    .validate(methodComponentContext, knnMethodConfigContext) == null;
            });
            this.methodComponents = methodComponents;
        }

        @Override
        public ValidationException validate(Object value, KNNMethodConfigContext knnMethodConfigContext) {
            ValidationException validationException = null;
            if (!(value instanceof MethodComponentContext)) {
                validationException = new ValidationException();
                validationException.addValidationError(
                    String.format("value is not an instance of for MethodComponentContext parameter [%s].", getName())
                );
                return validationException;
            }

            if (!validator.apply((MethodComponentContext) value, knnMethodConfigContext)) {
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
