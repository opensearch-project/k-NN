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
import org.opensearch.knn.common.KNNConstants;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * KNNMethod is used to define the structure of a method supported by a particular k-NN library. It is used to validate
 * the KNNMethodContext passed in by the user. It is also used to provide superficial string translations.
 */
public class KNNMethod {

    private final MethodComponent methodComponent;
    private final Set<SpaceType> spaces;

    /**
     * KNNMethod Constructor
     *
     * @param methodComponent top level method component that is compatible with the underlying library
     * @param spaces set of valid space types that the method supports
     */
    public KNNMethod(MethodComponent methodComponent, Set<SpaceType> spaces) {
        this.methodComponent = methodComponent;
        this.spaces = spaces;
    }

    /**
     * getMainMethodComponent
     *
     * @return mainMethodComponent
     */
    public MethodComponent getMethodComponent() {
        return methodComponent;
    }

    /**
     * Determines whether the provided space is supported for this method
     *
     * @param space to be checked
     * @return true if the space is supported; false otherwise
     */
    public boolean containsSpace(SpaceType space) {
        return spaces.contains(space);
    }

    /**
     * Get all valid spaces for this method
     *
     * @return spaces that can be used with this method
     */
    public Set<SpaceType> getSpaces() {
        return spaces;
    }

    /**
     * Validate that the configured KNNMethodContext is valid for this method
     *
     * @param knnMethodContext to be validated
     */
    public void validate(KNNMethodContext knnMethodContext) {
        if (!containsSpace(knnMethodContext.getSpaceType())) {
            throw new ValidationException();
        }

        methodComponent.validate(knnMethodContext.getMethodComponent());
    }

    /**
     * Parse knnMethodContext into a map that the library can use to configure the index
     *
     * @param knnMethodContext from which to generate map
     * @return KNNMethod as a map
     */
    public Map<String, Object> getAsMap(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameterMap = new HashMap<>(methodComponent.getAsMap(knnMethodContext.getMethodComponent()));
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
        public Builder addSpaces(SpaceType ...spaceTypes) {
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
