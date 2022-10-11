/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN940Codec;

import lombok.Builder;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNNFormatFacade;
import org.opensearch.knn.index.codec.KNNFormatFactory;

import java.util.Optional;

import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN94DefaultDelegate;

public class KNN940Codec extends FilterCodec {
    private static final String KNN940 = "KNN940Codec";
    private final KNNFormatFacade knnFormatFacade;
    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;

    /**
     * No arg constructor that uses Lucene94 as the delegate
     */
    public KNN940Codec() {
        this(createKNN94DefaultDelegate(), new KNN940PerFieldKnnVectorsFormat(Optional.empty()));
    }

    /**
     * Sole constructor. When subclassing this codec, create a no-arg ctor and pass the delegate codec
     * and a unique name to this ctor.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    @Builder
    protected KNN940Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat) {
        super(KNN940, delegate);
        knnFormatFacade = KNNFormatFactory.createKNN940Format(delegate);
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
