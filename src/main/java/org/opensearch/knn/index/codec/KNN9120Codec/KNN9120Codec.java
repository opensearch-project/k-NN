/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.Builder;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.KNNFormatFacade;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReadersSupplier;

/**
 * KNN Codec that wraps the Lucene Codec which is part of Lucene 9.12
 */
public class KNN9120Codec extends FilterCodec {
    private static final KNNCodecVersion VERSION = KNNCodecVersion.V_9_12_0;
    private final KNNFormatFacade knnFormatFacade;
    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;
    private final StoredFieldsFormat storedFieldsFormat;

    private final MapperService mapperService;

    /**
     * No arg constructor that uses Lucene99 as the delegate
     */
    public KNN9120Codec() {
        this(VERSION.getDefaultCodecDelegate(), VERSION.getPerFieldKnnVectorsFormat(), null);
    }

    /**
     * Sole constructor. When subclassing this codec, create a no-arg ctor and pass the delegate codec
     * and a unique name to this ctor.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    @Builder
    protected KNN9120Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat, MapperService mapperService) {
        super(VERSION.getCodecName(), delegate);
        knnFormatFacade = VERSION.getKnnFormatFacadeSupplier().apply(delegate);
        perFieldKnnVectorsFormat = knnVectorsFormat;
        this.mapperService = mapperService;
        this.storedFieldsFormat = getStoredFieldsFormat();
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

    @Override
    public StoredFieldsFormat storedFieldsFormat() {
        return storedFieldsFormat;
    }

    private StoredFieldsFormat getStoredFieldsFormat() {
        DerivedSourceReadersSupplier derivedSourceReadersSupplier = new DerivedSourceReadersSupplier((segmentReadState) -> {
            if (segmentReadState.fieldInfos.hasVectorValues()) {
                return knnVectorsFormat().fieldsReader(segmentReadState);
            }
            return null;
        }, (segmentReadState) -> {
            if (segmentReadState.fieldInfos.hasDocValues()) {
                return docValuesFormat().fieldsProducer(segmentReadState);
            }
            return null;

        }, (segmentReadState) -> {
            if (segmentReadState.fieldInfos.hasPostings()) {
                return postingsFormat().fieldsProducer(segmentReadState);
            }
            return null;

        }, (segmentReadState -> {
            if (segmentReadState.fieldInfos.hasNorms()) {
                return normsFormat().normsProducer(segmentReadState);
            }
            return null;
        }));
        return new DerivedSourceStoredFieldsFormat(delegate.storedFieldsFormat(), derivedSourceReadersSupplier, mapperService);
    }
}
