/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;

/**
 * This object provides additional context that the user does not provide when {@link KNNMethodContext} is
 * created via parsing. The values in this object need to be dynamically set and calling code needs to handle
 * the possibility that the values have not been set.
 */
@Setter
@Getter
@Builder
@AllArgsConstructor
public final class KNNMethodConfigContext {
    private VectorDataType vectorDataType;
    private Integer dimension;
    private Version versionCreated;

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        KNNMethodConfigContext other = (KNNMethodConfigContext) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(vectorDataType, other.vectorDataType);
        equalsBuilder.append(dimension, other.dimension);
        equalsBuilder.append(versionCreated, other.versionCreated);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(vectorDataType).append(dimension).append(versionCreated).toHashCode();
    }

    public static final KNNMethodConfigContext EMPTY = KNNMethodConfigContext.builder().build();
}
