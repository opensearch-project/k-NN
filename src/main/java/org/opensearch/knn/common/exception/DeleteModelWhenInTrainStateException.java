/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.OpenSearchException;
import org.opensearch.core.common.logging.LoggerMessageFormat;
import org.opensearch.rest.RestStatus;

/**
 * Exception thrown when a model is deleted while it is in the training state. The RestStatus associated with this
 * exception should be a {@link RestStatus#CONFLICT} because the request cannot be deleted due to the model being in
 * the training state.
 */
public class DeleteModelWhenInTrainStateException extends OpenSearchException {
    /**
     * Constructor
     *
     * @param msg detailed exception message
     * @param args arguments of the message
     */
    public DeleteModelWhenInTrainStateException(String msg, Object... args) {
        super(LoggerMessageFormat.format(msg, args));
    }

    @Override
    public RestStatus status() {
        return RestStatus.CONFLICT;
    }
}
