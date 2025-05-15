/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

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
        assert byteVector != null : "Byte vector is null";
        return byteVector;
    }

    public float[] getFloatVector() {
        assert floatVector != null : "Float vector is null";
        return floatVector;
    }

    public boolean hasFloatVector() {
        return floatVector != null && floatVector.length != 0;
    }

    public boolean hasByteVector() {
        return byteVector != null && byteVector.length != 0;
    }
}
