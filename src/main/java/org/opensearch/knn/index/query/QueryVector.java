/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import java.util.Arrays;

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
        if (hasByteVector()) {
            return byteVector;
        }
        throw new IllegalStateException("QueryVector does not have byte array");
    }

    public float[] getFloatVector() {
        if (hasFloatVector()) {
            return floatVector;
        }
        throw new IllegalStateException("QueryVector does not have float array");
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
            return Arrays.hashCode(floatVector);
        }
        if (hasByteVector()) {
            return Arrays.hashCode(byteVector);
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
            return Arrays.equals(floatVector, other.floatVector);
        }
        if (hasByteVector() && other.hasByteVector()) {
            return Arrays.equals(byteVector, other.byteVector);
        }
        return false;
    }

    @Override
    public String toString() {
        if (hasFloatVector()) {
            return "QueryVector[" + Arrays.toString(floatVector) + "]";
        }
        if (hasByteVector()) {
            return "QueryVector[" + Arrays.toString(byteVector) + "]";
        }
        throw new IllegalStateException("QueryVector has neither float nor byte array");
    }

    public int getLength() {
        if (hasFloatVector()) {
            return floatVector.length;
        }
        if (hasByteVector()) {
            return byteVector.length;
        }
        throw new IllegalStateException("QueryVector has neither float nor byte array");
    }
}
