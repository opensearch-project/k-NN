/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.KNN910Codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.KNNFormatFacade;

/**
 * Extends the Codec to support a new file format for KNN index
 * based on the mappings.
 *
 */
public final class KNN910Codec extends FilterCodec {
    private static final KNNCodecVersion VERSION = KNNCodecVersion.V_9_1_0;
    private final KNNFormatFacade knnFormatFacade;

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN910Codec() {
        this(VERSION.getDefaultCodecDelegate());
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     */
    public KNN910Codec(Codec delegate) {
        super(VERSION.getCodecName(), delegate);
        knnFormatFacade = VERSION.getKnnFormatFacadeSupplier().apply(delegate);
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return knnFormatFacade.docValuesFormat();
    }

    @Override
    public CompoundFormat compoundFormat() {
        return knnFormatFacade.compoundFormat();
    }
}
