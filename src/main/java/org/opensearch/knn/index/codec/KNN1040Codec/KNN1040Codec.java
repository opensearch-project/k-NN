/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.Builder;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReadersSupplier;

import java.util.Optional;

public class KNN1040Codec extends FilterCodec {
    private static final String NAME = "KNN1040Codec";
    public static final Codec DEFAULT_DELEGATE = new Lucene104Codec();
    private static final PerFieldKnnVectorsFormat DEFAULT_KNN_VECTOR_FORMAT = new KNN9120PerFieldKnnVectorsFormat(Optional.empty());

    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;
    private final StoredFieldsFormat storedFieldsFormat;

    private final MapperService mapperService;

    /**
     * No arg constructor that uses Lucene101Codec as the delegate
     */
    public KNN1040Codec() {
        this(DEFAULT_DELEGATE, DEFAULT_KNN_VECTOR_FORMAT, null);
    }

    /**
     * Sole constructor. When subclassing this codec, create a no-arg ctor and pass the delegate codec
     * and a unique name to this ctor.
     *
     * @param delegate codec that will perform all operations this codec does not override
     * @param knnVectorsFormat per field format for KnnVector
     */
    @Builder
    public KNN1040Codec(Codec delegate, PerFieldKnnVectorsFormat knnVectorsFormat, MapperService mapperService) {
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

        });
        return new KNN10010DerivedSourceStoredFieldsFormat(
            delegate.getName() /* codec name */,
            delegate.storedFieldsFormat(),
            derivedSourceReadersSupplier,
            mapperService
        );
    }
}
