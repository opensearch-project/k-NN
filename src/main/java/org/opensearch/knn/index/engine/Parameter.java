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
import java.util.function.Function;

/**
 * Parameter that can be set for a method component
 *
 * @param <T> Type parameter takes
 */
public abstract class Parameter<T> {
    @Getter
    private final String name;
    @Getter
    private final Function<KNNMethodConfigContext, T> defaultValueProvider;
    protected BiFunction<T, KNNMethodConfigContext, Boolean> validator;

    /**
     * Constructor
     *
     * @param name of the parameter
     * @param defaultValueProvider of the parameter based on the configuration context
     * @param validator used to validate a parameter value passed
     */
    public Parameter(
        String name,
        Function<KNNMethodConfigContext, T> defaultValueProvider,
        BiFunction<T, KNNMethodConfigContext, Boolean> validator
    ) {
        this.name = name;
        this.defaultValueProvider = defaultValueProvider;
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
        public BooleanParameter(
            String name,
            Function<KNNMethodConfigContext, Boolean> defaultValueProvider,
            BiFunction<Boolean, KNNMethodConfigContext, Boolean> validator
        ) {
            super(name, defaultValueProvider, validator);
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
        public IntegerParameter(
            String name,
            Function<KNNMethodConfigContext, Integer> defaultValueProvider,
            BiFunction<Integer, KNNMethodConfigContext, Boolean> validator
        ) {
            super(name, defaultValueProvider, validator);
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
        public DoubleParameter(
            String name,
            Function<KNNMethodConfigContext, Double> defaultValueProvider,
            BiFunction<Double, KNNMethodConfigContext, Boolean> validator
        ) {
            super(name, defaultValueProvider, validator);
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
        public StringParameter(
            String name,
            Function<KNNMethodConfigContext, String> defaultValueProvider,
            BiFunction<String, KNNMethodConfigContext, Boolean> validator
        ) {
            super(name, defaultValueProvider, validator);
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

        public MethodComponentContextParameter(
            String name,
            Function<KNNMethodConfigContext, MethodComponentContext> defaultValueProvider,
            Map<String, MethodComponent> methodComponents
        ) {
            super(name, defaultValueProvider, (methodComponentContext, knnMethodConfigContext) -> {
                MethodComponentContext resolvedMethodComponent = getMethodComponent(
                    knnMethodConfigContext,
                    methodComponentContext,
                    defaultValueProvider,
                    methodComponents
                ).resolveMethodComponentContext(knnMethodConfigContext, methodComponentContext);
                String resolvedMethodComponentName = resolvedMethodComponent.getName()
                    .orElseThrow(() -> new IllegalStateException("Resolved method shouldnt ever be null"));
                if (!methodComponents.containsKey(resolvedMethodComponentName)) {
                    return false;
                }
                return methodComponents.get(resolvedMethodComponentName).validate(methodComponentContext, knnMethodConfigContext) == null;
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
         * @param knnMethodConfigContext Configuration context
         * @param methodComponentContext Component context
         * @return MethodComponent that name maps to
         */
        public MethodComponent getMethodComponent(
            KNNMethodConfigContext knnMethodConfigContext,
            MethodComponentContext methodComponentContext
        ) {
            return getMethodComponent(knnMethodConfigContext, methodComponentContext, getDefaultValueProvider(), methodComponents);
        }

        private static MethodComponent getMethodComponent(
            KNNMethodConfigContext knnMethodConfigContext,
            MethodComponentContext methodComponentContext,
            Function<KNNMethodConfigContext, MethodComponentContext> defaultProvider,
            Map<String, MethodComponent> methodComponents
        ) {
            if (methodComponentContext != null && methodComponentContext.getName().isPresent()) {
                return methodComponents.get(methodComponentContext.getName().get());
            }
            MethodComponentContext resolvedMethodComponent = defaultProvider.apply(knnMethodConfigContext);
            return methodComponents.get(resolvedMethodComponent.getName().orElseThrow());
        }
    }
}
