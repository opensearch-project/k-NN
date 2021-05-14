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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * KNNMethod is used to define the structure of a method supported by a particular k-NN library. It is used to validate
 * the KNNMethodContext passed in by the user. It is also used to provide superficial string translations.
 */
public class KNNMethod {

    private MethodComponent methodComponent;
    private Set<SpaceType> spaces;

    /**
     * KNNMethod Constructor
     *
     * @param methodComponent top level method component that is compatible with the underlying library
     * @param spaces set of valid space types that the method supports
     */
    protected KNNMethod(MethodComponent methodComponent, Set<SpaceType> spaces) {
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
    public boolean hasSpace(SpaceType space) {
        return spaces.contains(space);
    }

    /**
     * Validate that the configured KNNMethodContext is valid for this method
     *
     * @param knnMethodContext to be validated
     */
    public void validate(KNNMethodContext knnMethodContext) {
        if (!hasSpace(knnMethodContext.getSpaceType())) {
            throw new ValidationException();
        }

        methodComponent.validate(knnMethodContext.getMethodComponent());
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
