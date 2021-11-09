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

package org.opensearch.knn.common.exception;

import java.util.List;

public class KNNInvalidIndicesException extends RuntimeException {

    private final List<String> invalidIndices;

    public KNNInvalidIndicesException(List<String> invalidIndices, String message) {
        super(message);
        this.invalidIndices = invalidIndices;
    }

    /**
     * Returns the Invalid Index
     *
     * @return invalid index name
     */
    public List<String> getInvalidIndices() {
        return invalidIndices;
    }

    @Override
    public String toString() {
        return "[KNN] " + String.join(",", invalidIndices) + ' ' + super.toString();
    }
}
