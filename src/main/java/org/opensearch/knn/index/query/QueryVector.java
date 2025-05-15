/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import java.util.Objects;

public class QueryVector {
    private float[] floatVector;
    private byte[] byteVector;

    public QueryVector(final float[] floatVector, final byte[] byteVector) {
        if (floatVector != null && byteVector != null) {
            throw new IllegalArgumentException(
                "Both float array and byte array cannot be set. QueryVector can accept either float or byte array"
            );
        }
        this.floatVector = floatVector;
        this.byteVector = byteVector;
    }

    public QueryVector(final float[] floatVector) {
        this.floatVector = floatVector;
    }

    public QueryVector(final byte[] byteVector) {
        this.byteVector = byteVector;
    }

    public byte[] getByteVector() {
        return byteVector;
    }

    public float[] getFloatVector() {
        return floatVector;
    }

    public boolean hasFloatVector() {
        return floatVector != null && floatVector.length != 0;
    }

    public boolean hasByteVector() {
        return byteVector != null && byteVector.length != 0;
    }

    @Override
    public int hashCode() {
        if (hasFloatVector()) {
            return Objects.hash(floatVector);
        }
        if (hasByteVector()) {
            return Objects.hash(byteVector);
        }
        return super.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        QueryVector other = (QueryVector) obj;
        if (hasFloatVector() && other.hasFloatVector()) {
            return Objects.equals(floatVector, other.floatVector);
        }
        if (hasByteVector() && other.hasByteVector()) {
            return Objects.equals(byteVector, other.byteVector);
        }
        return false;
    }
}
