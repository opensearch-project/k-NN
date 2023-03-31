/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.OpenSearchException;
import org.opensearch.common.logging.LoggerMessageFormat;
import org.opensearch.rest.RestStatus;

public class DeleteModelWhenInTrainStateException extends OpenSearchException {
    public DeleteModelWhenInTrainStateException(String msg, Object... args) {
        super(LoggerMessageFormat.format(msg, args));
    }

    @Override
    public RestStatus status() {
        return RestStatus.CONFLICT;
    }
}
