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

public class OutOfNativeMemoryException extends RuntimeException {

    /**
     * Constructor
     *
     * @param message Exception message to be appended.
     */
    public OutOfNativeMemoryException(String message) {
        super(message);
    }

    @Override
    public String toString() {
        return "[KNN] " + "Not enough room in native memory for allocation." + super.toString();
    }
}
