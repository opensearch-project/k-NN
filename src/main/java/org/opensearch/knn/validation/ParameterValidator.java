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

package org.opensearch.knn.validation;

import org.opensearch.common.Nullable;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class ParameterValidator {

    /**
     * A function which validates request parameters.
     * @param validParameters A set of valid parameters that can be requestParameters can be validated against
     * @param requestParameters parameters from the request
     * @return
     */
    @Nullable
    public static ValidationException validateParameters(
        final Map<String, Parameter<?>> validParameters,
        final Map<String, Object> requestParameters
    ) {

        if (validParameters == null) {
            throw new IllegalArgumentException("validParameters cannot be null");
        }

        if (requestParameters == null || requestParameters.isEmpty()) {
            return null;
        }

        final List<String> errorMessages = new ArrayList<>();
        for (Map.Entry<String, Object> parameter : requestParameters.entrySet()) {
            if (validParameters.containsKey(parameter.getKey())) {
                final ValidationException parameterValidation = validParameters.get(parameter.getKey()).validate(parameter.getValue());
                if (parameterValidation != null) {
                    errorMessages.addAll(parameterValidation.validationErrors());
                }
            } else {
                errorMessages.add("Unknown parameter '" + parameter.getKey() + "' found");
            }
        }

        if (errorMessages.isEmpty()) {
            return null;
        }

        final ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
    }
}
