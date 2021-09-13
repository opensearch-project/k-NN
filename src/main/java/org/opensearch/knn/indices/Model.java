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

    final private ModelInfo modelInfo;
    final private byte[] modelBlob;

    /**
     * Constructor
     *
     * @param modelInfo information about the model
     * @param modelBlob binary representation of model template index
     */
    public Model(ModelInfo modelInfo, byte[] modelBlob) {
        this.modelInfo = Objects.requireNonNull(modelInfo, "modelInfo must not be null");
        this.modelBlob = Objects.requireNonNull(modelBlob, "modelBlob must not be null");
    }

    /**
     * getter for model's info
     *
     * @return knnEngine
     */
    public ModelInfo getModelInfo() {
        return modelInfo;
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
        equalsBuilder.append(modelInfo, other.modelInfo);
        equalsBuilder.append(modelBlob, other.modelBlob);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(modelInfo).append(modelBlob).toHashCode();
    }
}
