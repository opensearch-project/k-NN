/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

public class TerminalIOExceptionTests extends KNNTestCase {

    public void testConstructorWithMessageAndCause() {
        String message = "Failed to write vector data";
        IOException cause = new IOException("disk full");
        TerminalIOException exception = new TerminalIOException(message, cause);

        assertEquals(message, exception.getMessage());
        assertSame(cause, exception.getCause());
    }

    public void testIsInstanceOfIOException() {
        TerminalIOException exception = new TerminalIOException("terminal error", new IOException("root cause"));

        assertTrue(exception instanceof IOException);
    }
}
