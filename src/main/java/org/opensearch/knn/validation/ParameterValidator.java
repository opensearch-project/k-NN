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
import org.opensearch.knn.index.util.EngineSpecificMethodContext;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

public final class ParameterValidator {

    /**
     * A function which validates request parameters.
     * @param validParameters A set of valid parameters that can be requestParameters can be validated against
     * @param requestParameters parameters from the request
     */
    @Nullable
    public static ValidationException validateParameters(
        final Map<String, Parameter<?>> validParameters,
        final Map<String, Object> requestParameters
    ) {
        return validateParameters(validParameters, requestParameters, null, null);
    }

    /**
     * A function which validates request parameters.
     * @param validParameters A set of valid parameters that can be requestParameters can be validated against
     * @param requestParameters parameters from the request
     * @param knnEngine The KNN engine
     * @param context The engine specific method context
     */
    @Nullable
    public static ValidationException validateParameters(
        final Map<String, Parameter<?>> validParameters,
        final Map<String, Object> requestParameters,
        final KNNEngine knnEngine,
        final EngineSpecificMethodContext.Context context
    ) {
        validateNonNullParameters(validParameters);

        if (requestParameters == null || requestParameters.isEmpty()) {
            return null;
        }

        List<String> errorMessages = new ArrayList<>();
        Set<String> checkedParameters = new HashSet<>();

        checkEngineSpecificErrors(knnEngine, context, errorMessages, checkedParameters);
        validateRequestParameters(validParameters, requestParameters, errorMessages, checkedParameters);

        return buildValidationException(errorMessages);
    }

    private static void validateNonNullParameters(Map<String, Parameter<?>> validParameters) {
        if (validParameters == null) {
            throw new IllegalArgumentException("validParameters cannot be null");
        }
    }

    private static void checkEngineSpecificErrors(
        KNNEngine knnEngine,
        EngineSpecificMethodContext.Context context,
        List<String> errorMessages,
        Set<String> checkedParameters
    ) {
        if (KNNEngine.LUCENE.equals(knnEngine) && context != null && context.isRadialSearch()) {
            errorMessages.add("ef_search is not supported for Lucene engine radial search");
            checkedParameters.add(METHOD_PARAMETER_EF_SEARCH);
        }
    }

    private static void validateRequestParameters(
        Map<String, Parameter<?>> validParameters,
        Map<String, Object> requestParameters,
        List<String> errorMessages,
        Set<String> checkedParameters
    ) {
        for (Map.Entry<String, Object> parameter : requestParameters.entrySet()) {
            if (checkedParameters.contains(parameter.getKey())) {
                continue;
            }
            if (validParameters.containsKey(parameter.getKey())) {
                final ValidationException parameterValidation = validParameters.get(parameter.getKey()).validate(parameter.getValue());
                if (parameterValidation != null) {
                    errorMessages.addAll(parameterValidation.validationErrors());
                }
            } else {
                errorMessages.add("Unknown parameter '" + parameter.getKey() + "' found");
            }
        }
    }

    @Nullable
    private static ValidationException buildValidationException(List<String> errorMessages) {
        if (errorMessages.isEmpty()) {
            return null;
        }

        final ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
    }
}
