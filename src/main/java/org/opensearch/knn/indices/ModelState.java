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

package org.opensearch.knn.indices;

import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;

import java.io.IOException;

/**
 * Defines life cycle state of the model
 */
public enum ModelState implements Writeable {
    TRAINING("training"),
    CREATED("created"),
    FAILED("failed");

    private final String name;

    ModelState(String name) {
        this.name = name;
    }

    /**
     * Get the name of the state the model is in.
     *
     * @return ModelState's name
     */
    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return getName();
    }

    @Override
    public void writeTo(StreamOutput streamOutput) throws IOException {
        streamOutput.writeString(this.getName());
    }

    /**
     * Return ModelState from StreamInput
     *
     * @param in StreamInput to read from
     * @return ModelState read from stream input
     */
    public static ModelState readFrom(StreamInput in) throws IOException {
        return getModelState(in.readString());
    }

    /**
     * Retrieve model state from the name
     *
     * @param name of the state
     * @return ModelState
     */
    public static ModelState getModelState(String name) {
        if (TRAINING.getName().equals(name)) {
            return TRAINING;
        }

        if (CREATED.getName().equals(name)) {
            return CREATED;
        }

        if (FAILED.getName().equals(name)) {
            return FAILED;
        }

        throw new IllegalArgumentException("Unable to find model state: \"" + name + "\"");
    }
}
