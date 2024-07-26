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

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.training.VectorSpaceInfo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * KNNMethod is used to define the structure of a method supported by a particular k-NN library. It is used to validate
 * the KNNMethodContext passed in by the user. It is also used to provide superficial string translations.
 */
@AllArgsConstructor
@Getter
public class KNNMethod {

    private final MethodComponent methodComponent;
    private final Set<SpaceType> spaces;

    /**
     * Determines whether the provided space is supported for this method
     *
     * @param space to be checked
     * @return true if the space is supported; false otherwise
     */
    public boolean isSpaceTypeSupported(SpaceType space) {
        return spaces.contains(space);
    }

    /**
     * Validate that the configured KNNMethodContext is valid for this method
     *
     * @param knnMethodContext to be validated
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validate(KNNMethodContext knnMethodContext) {
        List<String> errorMessages = new ArrayList<>();
        if (!isSpaceTypeSupported(knnMethodContext.getSpaceType())) {
            errorMessages.add(
                String.format(
                    Locale.ROOT,
                    "\"%s\" with \"%s\" configuration does not support space type: " + "\"%s\".",
                    this.methodComponent.getName(),
                    knnMethodContext.getKnnEngine().getName().toLowerCase(Locale.ROOT),
                    knnMethodContext.getSpaceType().getValue()
                )
            );
        }

        ValidationException methodValidation = methodComponent.validate(knnMethodContext.getMethodComponentContext());
        if (methodValidation != null) {
            errorMessages.addAll(methodValidation.validationErrors());
        }

        if (errorMessages.isEmpty()) {
            return null;
        }

        ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
    }

    /**
     * Validate that the configured KNNMethodContext is valid for this method, using additional data not present in the method context
     *
     * @param knnMethodContext to be validated
     * @param vectorSpaceInfo additional data not present in the method context
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validateWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo) {
        List<String> errorMessages = new ArrayList<>();
        if (!isSpaceTypeSupported(knnMethodContext.getSpaceType())) {
            errorMessages.add(
                String.format(
                    Locale.ROOT,
                    "\"%s\" with \"%s\" configuration does not support space type: " + "\"%s\".",
                    this.methodComponent.getName(),
                    knnMethodContext.getKnnEngine().getName().toLowerCase(Locale.ROOT),
                    knnMethodContext.getSpaceType().getValue()
                )
            );
        }

        ValidationException methodValidation = methodComponent.validateWithData(
            knnMethodContext.getMethodComponentContext(),
            vectorSpaceInfo
        );
        if (methodValidation != null) {
            errorMessages.addAll(methodValidation.validationErrors());
        }

        if (errorMessages.isEmpty()) {
            return null;
        }

        ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errorMessages);
        return validationException;
    }

    /**
     * returns whether training is required or not
     *
     * @param knnMethodContext context to check if training is required on
     * @return true if training is required; false otherwise
     */
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        return methodComponent.isTrainingRequired(knnMethodContext.getMethodComponentContext());
    }

    /**
     * Returns the estimated overhead of the method in KB
     *
     * @param knnMethodContext context to estimate overhead
     * @param dimension dimension to make estimate with
     * @return estimate overhead in KB
     */
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
        return methodComponent.estimateOverheadInKB(knnMethodContext.getMethodComponentContext(), dimension);
    }

    /**
     * Parse knnMethodContext into a map that the library can use to configure the index
     *
     * @param knnMethodContext from which to generate map
     * @return KNNMethod as a map
     */
    public Map<String, Object> getAsMap(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameterMap = new HashMap<>(methodComponent.getAsMap(knnMethodContext.getMethodComponentContext()));
        parameterMap.put(KNNConstants.SPACE_TYPE, knnMethodContext.getSpaceType().getValue());
        return parameterMap;
    }

    /**
     * Builder for KNNMethod
     */
    public static class Builder {

        private MethodComponent methodComponent;
        private Set<SpaceType> spaces;

        /**
         * Method to get a Builder instance
         *
         * @param methodComponent top level method component for the method
         * @return Builder instance
         */
        public static Builder builder(MethodComponent methodComponent) {
            return new Builder(methodComponent);
        }

        private Builder(MethodComponent methodComponent) {
            this.methodComponent = methodComponent;
            this.spaces = new HashSet<>();
        }

        /**
         * Add spaces to KNNMethod
         *
         * @param spaceTypes to be added
         * @return Builder
         */
        public Builder addSpaces(SpaceType... spaceTypes) {
            spaces.addAll(Arrays.asList(spaceTypes));
            return this;
        }

        /**
         * Build KNNMethod from builder
         *
         * @return KNNMethod initialized from builder
         */
        public KNNMethod build() {
            return new KNNMethod(methodComponent, spaces);
        }
    }
}
