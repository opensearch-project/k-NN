/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN87Codec;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.backward_codecs.lucene87.Lucene87Codec;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;

/**
 * Extends the Codec to support a new file format for KNN index
 * based on the mappings.
 *
 */
public final class KNN87Codec extends FilterCodec {

    private final DocValuesFormat docValuesFormat;
    private final CompoundFormat compoundFormat;

    public static final String KNN_87 = "KNN87Codec";

    /**
     * No arg constructor that uses Lucene87 as the delegate
     */
    public KNN87Codec() {
        this(new Lucene87Codec());
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     */
    public KNN87Codec(Codec delegate) {
        super(KNN_87, delegate);
        // Note that DocValuesFormat can use old Codec's DocValuesFormat. For instance Lucene84 uses Lucene80
        // DocValuesFormat. Refer to defaultDVFormat in LuceneXXCodec.java to find out which version it uses
        this.docValuesFormat = new KNN80DocValuesFormat(delegate.docValuesFormat());
        this.compoundFormat = new KNN80CompoundFormat(delegate.compoundFormat());
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return this.docValuesFormat;
    }

    @Override
    public CompoundFormat compoundFormat() {
        return this.compoundFormat;
    }
}
