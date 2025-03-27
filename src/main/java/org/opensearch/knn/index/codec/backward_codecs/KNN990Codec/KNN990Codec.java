/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN990Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;

import java.util.Optional;

/**
 * KNN Codec that wraps the Lucene Codec which is part of Lucene 9.9
 */
public class KNN990Codec extends FilterCodec {
    private static final String NAME = "KNN990Codec";
    private static final Codec DEFAULT_DELEGATE = new Lucene99Codec();
    private static final PerFieldKnnVectorsFormat DEFAULT_KNN_VECTOR_FORMAT = new KNN990PerFieldKnnVectorsFormat(Optional.empty());

    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;

    /**
     * No arg constructor that uses Lucene99 as the delegate
     */
    public KNN990Codec() {
        this(DEFAULT_DELEGATE, DEFAULT_KNN_VECTOR_FORMAT);
    }

    /**
     * Sole constructor. When subclassing this codec, create a no-arg ctor and pass the delegate codec
     * and a unique name to this ctor.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    private KNN990Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat) {
        super(NAME, delegate);
        perFieldKnnVectorsFormat = knnVectorsFormat;
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return new KNN80DocValuesFormat(delegate.docValuesFormat());
    }

    @Override
    public CompoundFormat compoundFormat() {
        return new KNN80CompoundFormat(delegate.compoundFormat());
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return perFieldKnnVectorsFormat;
    }
}
