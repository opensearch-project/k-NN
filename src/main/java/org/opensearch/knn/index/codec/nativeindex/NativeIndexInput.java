/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import java.io.IOException;

import org.apache.lucene.store.IndexInput;

/**
 * NativeIndexInput allows us to pass down single IndexInput for drived index inputs that are cloned from sourc index
 * inputs through IndexInput.slice method. When cloned/sliced index input is closed, source should be closed as well.
 * This index input extension allows passing both index inputs down to consumer so when close is called, both
 * index inputs are closed. See NativeIndexReader for usage of this class.
 */
public class NativeIndexInput extends IndexInput {

    private final IndexInput sliceSourceIndexInput;
    private final IndexInput indexInput;

    public NativeIndexInput(final String name, final IndexInput sliceSourceIndexInput, final IndexInput indexInput) {
        super(name);
        this.sliceSourceIndexInput = sliceSourceIndexInput;
        this.indexInput = indexInput;
    }

    @Override
    public void close() throws IOException {
        this.indexInput.close();
        this.sliceSourceIndexInput.close();
    }

    @Override
    public long getFilePointer() {
        return this.indexInput.getFilePointer();
    }

    @Override
    public void seek(long l) throws IOException {
        this.indexInput.seek(l);
    }

    @Override
    public long length() {
        return this.indexInput.length();
    }

    @Override
    public IndexInput slice(String s, long l, long l1) throws IOException {
        return this.indexInput.slice(s, l, l1);
    }

    @Override
    public byte readByte() throws IOException {
        return this.indexInput.readByte();
    }

    @Override
    public void readBytes(byte[] bytes, int i, int i1) throws IOException {
        this.indexInput.readBytes(bytes, i, i1);
    }
}
