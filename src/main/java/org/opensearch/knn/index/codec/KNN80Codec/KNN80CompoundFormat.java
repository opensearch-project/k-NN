/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import org.apache.lucene.backward_codecs.lucene50.Lucene50CompoundFormat;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;

import java.io.IOException;

/**
 * Class to encode/decode compound file
 */
public class KNN80CompoundFormat extends CompoundFormat {

    private final CompoundFormat delegate;

    public KNN80CompoundFormat() {
        this.delegate = new Lucene50CompoundFormat();
    }

    /**
     * Constructor that takes a delegate to handle non-overridden methods
     *
     * @param delegate CompoundFormat that will handle non-overridden methods
     */
    public KNN80CompoundFormat(CompoundFormat delegate) {
        this.delegate = delegate;
    }

    @Override
    public CompoundDirectory getCompoundReader(Directory dir, SegmentInfo si) throws IOException {
        return new KNN80CompoundDirectory(delegate.getCompoundReader(dir, si), dir);
    }

    @Override
    public void write(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        delegate.write(dir, si, context);
    }
}
