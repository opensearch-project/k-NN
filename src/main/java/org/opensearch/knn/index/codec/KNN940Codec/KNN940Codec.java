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
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.KNNFormatFacade;

public class KNN940Codec extends FilterCodec {
    private static final KNNCodecVersion VERSION = KNNCodecVersion.V_9_4_0;
    private final KNNFormatFacade knnFormatFacade;
    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;

    /**
     * No arg constructor that uses Lucene94 as the delegate
     */
    public KNN940Codec() {
        this(VERSION.getDefaultCodecDelegate(), VERSION.getPerFieldKnnVectorsFormat());
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
