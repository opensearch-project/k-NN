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

package org.opensearch.knn.index.query.request;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.core.ParseField;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.index.query.KNNQueryBuilder.EF_SEARCH_FIELD;

/**
 * MethodParameters are engine and algorithm related parameters that clients can pass in knn query
 * This enum holds metadata which helps parse and have basic validation related to MethodParameter
 */
@Getter
@RequiredArgsConstructor
public enum MethodParameter {

    // TODO: change the version to 2.16 when merging into 2.x
    EF_SEARCH(METHOD_PARAMETER_EF_SEARCH, Version.CURRENT, EF_SEARCH_FIELD) {
        @Override
        public Integer parse(Object value) {
            try {
                return Integer.parseInt(String.valueOf(value));
            } catch (final NumberFormatException e) {
                throw new IllegalArgumentException(METHOD_PARAMETER_EF_SEARCH + " value must be an integer");
            }
        }

        @Override
        public ValidationException validate(Object value) {
            final Integer ef = parse(value);
            if (ef != null && ef > 0) {
                return null;
            }
            ;
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(METHOD_PARAMETER_EF_SEARCH + " should be greater than 0");
            return validationException;
        }
    };

    private final String name;
    private final Version version;
    private final ParseField parseField;

    private static Map<String, MethodParameter> PARAMETERS_DIR;

    public abstract <T> T parse(Object value);

    // These are preliminary validations on rest layer
    public abstract ValidationException validate(Object value);

    public static MethodParameter enumOf(final String name) {
        if (PARAMETERS_DIR == null) {
            PARAMETERS_DIR = new HashMap<>();
            for (final MethodParameter methodParameter : MethodParameter.values()) {
                PARAMETERS_DIR.put(methodParameter.name, methodParameter);
            }
        }
        return PARAMETERS_DIR.get(name);
    }
}
