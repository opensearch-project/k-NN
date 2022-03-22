/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.KNN91Codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.opensearch.knn.index.codec.KNNFormatFacade;
import org.opensearch.knn.index.codec.KNNFormatFactory;

import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN91DefaultDelegate;

/**
 * Extends the Codec to support a new file format for KNN index
 * based on the mappings.
 *
 */
public final class KNN91Codec extends FilterCodec {

    public static final String KNN_91 = "KNN91Codec";
    private KNNFormatFacade knnFormatFacade;

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN91Codec() {
        this(createKNN91DefaultDelegate());
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     */
    public KNN91Codec(Codec delegate) {
        super(KNN_91, delegate);
        knnFormatFacade = KNNFormatFactory.createKNN91Format(delegate);
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
