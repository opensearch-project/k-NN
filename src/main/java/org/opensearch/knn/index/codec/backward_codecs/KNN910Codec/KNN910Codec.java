/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.backward_codecs.KNN910Codec;

import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;

/**
 * Extends the Codec to support a new file format for KNN index
 * based on the mappings.
 *
 */
public final class KNN910Codec extends FilterCodec {

    private static final String NAME = "KNN910Codec";
    private static final Codec DEFAULT_DELEGATE = new Lucene91Codec();

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN910Codec() {
        this(DEFAULT_DELEGATE);
    }

    private KNN910Codec(Codec delegate) {
        super(NAME, delegate);
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return new KNN80DocValuesFormat(delegate.docValuesFormat());
    }

    @Override
    public CompoundFormat compoundFormat() {
        return new KNN80CompoundFormat(delegate.compoundFormat());
    }
}
