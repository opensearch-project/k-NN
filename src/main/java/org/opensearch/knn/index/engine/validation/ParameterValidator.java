/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.validation;

import org.opensearch.common.Nullable;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class ParameterValidator {

    /**
     * A function which validates request parameters.
     *
     * @param validParameters A set of valid parameters that can be requestParameters can be validated against
     * @param requestParameters parameters from the request
     * @param knnMethodConfigContext context of the knn method
     * @return ValidationException if there are any validation errors, null otherwise
     */
    @Nullable
    public static ValidationException validateParameters(
        final Map<String, Parameter<?>> validParameters,
        final Map<String, Object> requestParameters,
        KNNMethodConfigContext knnMethodConfigContext
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
                final ValidationException parameterValidation = validParameters.get(parameter.getKey())
                    .validate(parameter.getValue(), knnMethodConfigContext);
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
