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

import org.opensearch.OpenSearchException;
import org.opensearch.core.common.logging.LoggerMessageFormat;

/**
 * Exception thrown when a model is deleted while it is in use by an index.
 */
public class DeleteModelWhenInUseException extends OpenSearchException {
    /**
     * Constructor
     *
     * @param msg detailed exception message
     * @param args arguments of the message
     */
    public DeleteModelWhenInUseException(String msg, Object... args) {
        super(LoggerMessageFormat.format(msg, args));
    }
}
