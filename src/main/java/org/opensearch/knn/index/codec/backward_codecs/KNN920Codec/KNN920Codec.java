/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.backward_codecs.KNN920Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.backward_codecs.lucene92.Lucene92Codec;
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
 * KNN codec that is based on Lucene92 codec
 */
@Log4j2
public final class KNN920Codec extends FilterCodec {
    private static final String NAME = "KNN920Codec";
    private static final Codec DEFAULT_DELEGATE = new Lucene92Codec();
    private static final PerFieldKnnVectorsFormat DEFAULT_KNN_VECTOR_FORMAT = new KNN920PerFieldKnnVectorsFormat(Optional.empty());

    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN920Codec() {
        this(DEFAULT_DELEGATE, DEFAULT_KNN_VECTOR_FORMAT);
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    private KNN920Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat) {
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
