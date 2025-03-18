/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.apache.lucene.backward_codecs.lucene912.Lucene912Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;

import java.util.Optional;

/**
 * KNN Codec that wraps the Lucene Codec which is part of Lucene 9.12
 */
public class KNN9120Codec extends FilterCodec {
    private static final String NAME = "KNN9120Codec";
    private static final Codec DEFAULT_DELEGATE = new Lucene912Codec();
    private static final PerFieldKnnVectorsFormat DEFAULT_KNN_VECTOR_FORMAT = new KNN9120PerFieldKnnVectorsFormat(Optional.empty());

    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;
    private final StoredFieldsFormat storedFieldsFormat;

    private final MapperService mapperService;

    /**
     * No arg constructor that uses Lucene99 as the delegate
     */
    public KNN9120Codec() {
        this(DEFAULT_DELEGATE, DEFAULT_KNN_VECTOR_FORMAT, null);
    }

    /**
     * Sole constructor. When subclassing this codec, create a no-arg ctor and pass the delegate codec
     * and a unique name to this ctor.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    private KNN9120Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat, MapperService mapperService) {
        super(NAME, delegate);
        perFieldKnnVectorsFormat = knnVectorsFormat;
        this.mapperService = mapperService;
        this.storedFieldsFormat = getStoredFieldsFormat();
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

    @Override
    public StoredFieldsFormat storedFieldsFormat() {
        return storedFieldsFormat;
    }

    private StoredFieldsFormat getStoredFieldsFormat() {
        KNN9120DerivedSourceReadersSupplier derivedSourceReadersSupplier = new KNN9120DerivedSourceReadersSupplier((segmentReadState) -> {
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
        return new KNN9120DerivedSourceStoredFieldsFormat(delegate.storedFieldsFormat(), derivedSourceReadersSupplier, mapperService);
    }
}
