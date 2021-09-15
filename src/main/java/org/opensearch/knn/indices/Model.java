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

import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;
import org.opensearch.common.Nullable;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class Model {

    private ModelMetadata modelMetadata;
    private AtomicReference<byte[]> modelBlob;

    /**
     * Constructor
     *
     * @param modelMetadata metadata about the model
     * @param modelBlob binary representation of model template index. Can be null if model is not yet in CREATED state.
     */
    public Model(ModelMetadata modelMetadata, @Nullable byte[] modelBlob) {
        this.modelMetadata = Objects.requireNonNull(modelMetadata, "modelMetadata must not be null");

        if (ModelState.CREATED.equals(this.modelMetadata.getState()) && modelBlob == null) {
            throw new IllegalArgumentException("Model blob cannot be null when model metadata says model is created");
        }

        this.modelBlob = new AtomicReference<>(modelBlob);
    }

    /**
     * getter for model's metadata
     *
     * @return knnEngine
     */
    public ModelMetadata getModelMetadata() {
        return modelMetadata;
    }

    /**
     * getter for model's binary blob
     *
     * @return modelBlob
     */
    public byte[] getModelBlob() {
        return modelBlob.get();
    }

    /**
     * getter for model's length
     *
     * @return length of model blob
     */
    public int getLength() {
        if (getModelBlob() == null) {
            return 0;
        }
        return getModelBlob().length;
    }

    /**
     * Sets model blob to new value
     *
     * @param modelBlob updated model blob
     */
    public synchronized void setModelBlob(byte[] modelBlob) {
        this.modelBlob = new AtomicReference<>(Objects.requireNonNull(modelBlob, "model blob cannot be updated to null"));
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        Model other = (Model) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(modelMetadata, other.modelMetadata);
        equalsBuilder.append(getModelBlob(), other.getModelBlob());

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(modelMetadata).append(getModelBlob()).toHashCode();
    }
}
