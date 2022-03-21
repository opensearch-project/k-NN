/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN91Codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.lucene91.Lucene91Codec;

/**
 * Extends the Codec to support a new file format for KNN index
 * based on the mappings.
 *
 */
public final class KNN91Codec extends FilterCodec {

    private final DocValuesFormat docValuesFormat;
    private final CompoundFormat compoundFormat;

    public static final String KNN_91 = "KNN91Codec";

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN91Codec() {
        this(new Lucene91Codec());
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     */
    public KNN91Codec(Codec delegate) {
        super(KNN_91, delegate);
        this.docValuesFormat = new KNN91DocValuesFormat(delegate.docValuesFormat());
        this.compoundFormat = new KNN91CompoundFormat(delegate.compoundFormat());
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
