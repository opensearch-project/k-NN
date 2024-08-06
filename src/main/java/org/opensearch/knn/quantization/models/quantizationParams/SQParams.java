/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationParams;

import org.opensearch.knn.quantization.enums.QuantizationType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.Objects;

/**
 * The SQParams class represents the parameters specific to scalar quantization (SQ).
 * This class implements the QuantizationParams interface and includes the type of scalar quantization.
 */
public class SQParams implements QuantizationParams {
    private final ScalarQuantizationType sqType;

    /**
     * Constructs an SQParams instance with the specified scalar quantization type.
     *
     * @param sqType The specific type of scalar quantization (e.g., ONE_BIT, TWO_BIT, FOUR_BIT).
     */
    public SQParams(final ScalarQuantizationType sqType) {
        this.sqType = sqType;
    }

    /**
     * Returns the quantization type associated with these parameters.
     *
     * @return The quantization type, always VALUE_QUANTIZATION for SQParams.
     */
    @Override
    public QuantizationType getQuantizationType() {
        return QuantizationType.VALUE;
    }

    /**
     * Returns the scalar quantization type.
     *
     * @return The specific scalar quantization type.
     */
    public ScalarQuantizationType getSqType() {
        return sqType;
    }

    /**
     * Provides a unique type identifier for the SQParams, combining the quantization type and SQ type.
     * This identifier is useful for distinguishing between different configurations of scalar quantization parameters.
     *
     * @return A string representing the unique type identifier.
     */
    @Override
    public String getTypeIdentifier() {
        return getQuantizationType().name() + "_" + sqType.name();
    }

    /**
     * Compares this object to the specified object. The result is true if and only if the argument is not null and is
     * an SQParams object that represents the same scalar quantization type.
     *
     * @param o The object to compare this SQParams against.
     * @return true if the given object represents an SQParams equivalent to this instance, false otherwise.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SQParams sqParams = (SQParams) o;
        return sqType == sqParams.sqType;
    }

    /**
     * Returns a hash code value for this SQParams instance.
     *
     * @return A hash code value for this SQParams instance.
     */
    @Override
    public int hashCode() {
        return Objects.hash(sqType);
    }
}
