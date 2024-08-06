/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.Getter;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.Set;

public class KNN80CompoundDirectory extends CompoundDirectory {

    @Getter
    private CompoundDirectory delegate;
    @Getter
    private Directory dir;
    public KNN80CompoundDirectory(CompoundDirectory delegate, Directory dir) {
        this.delegate = delegate;
        this.dir = dir;
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public String[] listAll() throws IOException {
        return delegate.listAll();
    }

    @Override
    public long fileLength(String name) throws IOException {
        return delegate.fileLength(name);
    }

    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
        return delegate.openInput(name, context);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public Set<String> getPendingDeletions() throws IOException {
        return delegate.getPendingDeletions();
    }

}
