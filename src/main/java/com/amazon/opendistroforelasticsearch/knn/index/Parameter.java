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

package com.amazon.opendistroforelasticsearch.knn.index;

import org.opensearch.common.ValidationException;

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
}
