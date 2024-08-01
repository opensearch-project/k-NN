/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

/**
 * Interface representing an encoder. An encoder generally refers to a vector quantizer.
 */
public interface Encoder {
    /**
     * The name of the encoder does not have to be unique. Howevwer, when using within a method, there cannot be
     * 2 encoders with the same name.
     *
     * @return Name of the encoder
     */
    String getName();

    /**
     *
     * @return Method component associated with the encoder
     */
    MethodComponent getMethodComponent();
}
