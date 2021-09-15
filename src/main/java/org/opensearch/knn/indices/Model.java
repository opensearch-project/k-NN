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

import java.util.Objects;

public class Model {

    final private ModelMetadata modelMetadata;
    final private byte[] modelBlob;

    /**
     * Constructor
     *
     * @param modelMetadata metadata about the model
     * @param modelBlob binary representation of model template index
     */
    public Model(ModelMetadata modelMetadata, byte[] modelBlob) {
        this.modelMetadata = Objects.requireNonNull(modelMetadata, "modelMetadata must not be null");
        this.modelBlob = Objects.requireNonNull(modelBlob, "modelBlob must not be null");
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
        return modelBlob;
    }

    /**
     * getter for model's length
     *
     * @return length of model blob
     */
    public int getLength() {
        return modelBlob.length;
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
        equalsBuilder.append(modelBlob, other.modelBlob);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(modelMetadata).append(modelBlob).toHashCode();
    }
}
