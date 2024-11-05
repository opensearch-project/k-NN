/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class RaisingIOExceptionIndexInput extends IndexInput {
    public RaisingIOExceptionIndexInput() {
        super(RaisingIOExceptionIndexInput.class.getSimpleName());
    }

    @Override
    public void close() throws IOException {
        throw new IOException("RaisingIOExceptionIndexInput::readBytes failed.");
    }

    @Override
    public long getFilePointer() {
        throw new RuntimeException("RaisingIOExceptionIndexInput::readBytes failed.");
    }

    @Override
    public void seek(long l) throws IOException {
        throw new IOException("RaisingIOExceptionIndexInput::readBytes failed.");
    }

    @Override
    public long length() {
        return 0;
    }

    @Override
    public IndexInput slice(String s, long l, long l1) throws IOException {
        throw new IOException("RaisingIOExceptionIndexInput::readBytes failed.");
    }

    @Override
    public byte readByte() throws IOException {
        throw new IOException("RaisingIOExceptionIndexInput::readBytes failed.");
    }

    @Override
    public void readBytes(byte[] bytes, int i, int i1) throws IOException {
        throw new IOException("RaisingIOExceptionIndexInput::readBytes failed.");
    }
}
