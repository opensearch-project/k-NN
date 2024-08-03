/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

import lombok.Getter;

/**
 * The ScalarQuantizationType enum defines the various scalar quantization types that can be used
 * for vector quantization. Each type corresponds to a different bit-width representation of the quantized values.
 *
 * <p>
 * Future Developers: If you change the name of any enum constant, do not change its associated value.
 * Serialization and deserialization depend on these values to maintain compatibility.
 * </p>
 */
@Getter
public enum ScalarQuantizationType {
    /**
     * ONE_BIT quantization uses a single bit per coordinate.
     */
    ONE_BIT(1),

    /**
     * TWO_BIT quantization uses two bits per coordinate.
     */
    TWO_BIT(2),

    /**
     * FOUR_BIT quantization uses four bits per coordinate.
     */
    FOUR_BIT(4);

    private final int id;

    /**
     * Constructs a ScalarQuantizationType with the specified ID.
     *
     * @param id the ID representing the quantization type.
     */
    ScalarQuantizationType(int id) {
        this.id = id;
    }
}
