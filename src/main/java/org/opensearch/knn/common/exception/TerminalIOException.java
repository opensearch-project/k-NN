/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.knn.index.codec.nativeindex.remote.DefaultVectorRepositoryAccessor;

import java.io.IOException;

/**
 * Custom exception class to distinguish between terminal IOExceptions (in the case of writing to index output,
 * for example) and non-terminal (e.g. reading from S3). See {@link DefaultVectorRepositoryAccessor#readFromRepository}.
 * Useful in GPU builds for determining whether to fall back to CPU build or terminate build altogether.
 */
public class TerminalIOException extends IOException {
    public TerminalIOException(String message, Throwable cause) {
        super(message, cause);
    }
}
