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

    private T defaultValue;
    protected Predicate<T> validator;

    /**
     * Constructor
     *
     * @param defaultValue of the parameter
     * @param validator used to validate a parameter value passed
     */
    public Parameter(T defaultValue, Predicate<T> validator) {
        this.defaultValue = defaultValue;
        this.validator = validator;
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
     */
    public abstract void validate(Object value);

    /**
     * Integer method parameter
     */
    public static class IntegerParameter extends Parameter<Integer> {
        public IntegerParameter(Integer defaultValue, Predicate<Integer> validator)
        {
            super(defaultValue, validator);
        }

        @Override
        public void validate(Object value) {
            if (!(value instanceof Integer) || !validator.test((Integer) value)) {
                throw new ValidationException();
            }
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
         * @param defaultValue value to assign this parameter if it is not set
         * @param methodComponents valid components that the MethodComponentContext can map to
         */
        public MethodComponentContextParameter(MethodComponentContext defaultValue,
                                               Map<String, MethodComponent> methodComponents) {
            super(defaultValue, methodComponentContext -> {
                if (!methodComponents.containsKey(methodComponentContext.getName())) {
                    return false;
                }

                try {
                    methodComponents.get(methodComponentContext.getName()).validate(methodComponentContext);
                } catch (ValidationException ex) {
                    return false;
                }

                return true;
            });
            this.methodComponents = methodComponents;
        }

        @Override
        public void validate(Object value) {
            if (!(value instanceof MethodComponentContext) || !validator.test((MethodComponentContext) value)) {
                throw new ValidationException();
            }
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
