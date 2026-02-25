/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.OpenSearchException;
import org.opensearch.core.common.logging.LoggerMessageFormat;
import org.opensearch.core.rest.RestStatus;

/**
 * Exception thrown when a model is deleted while it is in the training state or in use by an index. The RestStatus associated with this
 * exception should be a {@link RestStatus#CONFLICT} because the request cannot be deleted due to the model being in
 * the training state or in use by an index.
 */
public class DeleteModelException extends OpenSearchException {
    /**
     * Constructor
     *
     * @param msg detailed exception message
     * @param args arguments of the message
     */
    public DeleteModelException(String msg, Object... args) {
        super(LoggerMessageFormat.format(msg, args));
    }

    @Override
    public RestStatus status() {
        return RestStatus.CONFLICT;
    }
}
