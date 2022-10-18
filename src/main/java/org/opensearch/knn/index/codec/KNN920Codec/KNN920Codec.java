/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.KNN920Codec;

import lombok.Builder;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.KNNFormatFacade;

/**
 * KNN codec that is based on Lucene92 codec
 */
@Log4j2
public final class KNN920Codec extends FilterCodec {
    private static final KNNCodecVersion VERSION = KNNCodecVersion.V_9_2_0;
    private final KNNFormatFacade knnFormatFacade;
    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN920Codec() {
        this(VERSION.getDefaultCodecDelegate(), VERSION.getPerFieldKnnVectorsFormat());
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    @Builder
    public KNN920Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat) {
        super(VERSION.getCodecName(), delegate);
        knnFormatFacade = VERSION.getKnnFormatFacadeSupplier().apply(delegate);
        perFieldKnnVectorsFormat = knnVectorsFormat;
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return knnFormatFacade.docValuesFormat();
    }

    @Override
    public CompoundFormat compoundFormat() {
        return knnFormatFacade.compoundFormat();
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return perFieldKnnVectorsFormat;
    }
}
