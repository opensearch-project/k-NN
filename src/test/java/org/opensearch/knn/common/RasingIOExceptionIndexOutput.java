/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

public class RasingIOExceptionIndexOutput extends IndexOutput {
    public RasingIOExceptionIndexOutput() {
        super("Always throws IOException", RasingIOExceptionIndexOutput.class.getSimpleName());
    }

    @Override
    public void close() throws IOException {
        throw new IOException("RaiseIOExceptionIndexInput::close failed.");
    }

    @Override
    public long getFilePointer() {
        throw new RuntimeException("RaiseIOExceptionIndexInput::getFilePointer failed.");
    }

    @Override
    public long getChecksum() throws IOException {
        throw new IOException("RaiseIOExceptionIndexInput::getChecksum failed.");
    }

    @Override
    public void writeByte(byte b) throws IOException {
        throw new IOException("RaiseIOExceptionIndexInput::writeByte failed.");
    }

    @Override
    public void writeBytes(byte[] bytes, int i, int i1) throws IOException {
        throw new IOException("RaiseIOExceptionIndexInput::writeBytes failed.");
    }
}
