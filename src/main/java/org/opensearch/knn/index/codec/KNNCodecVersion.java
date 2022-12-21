/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.backward_codecs.lucene92.Lucene92Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene94.Lucene94Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;
import org.opensearch.knn.index.codec.KNN910Codec.KNN910Codec;
import org.opensearch.knn.index.codec.KNN920Codec.KNN920Codec;
import org.opensearch.knn.index.codec.KNN920Codec.KNN920PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN940Codec.KNN940Codec;
import org.opensearch.knn.index.codec.KNN940Codec.KNN940PerFieldKnnVectorsFormat;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Abstraction for k-NN codec version, aggregates all details for specific version such as codec name, corresponding
 * Lucene codec, formats including one for k-NN vector etc.
 */
@AllArgsConstructor
@Getter
public enum KNNCodecVersion {

    V_9_1_0(
        "KNN910Codec",
        new Lucene91Codec(),
        null,
        (delegate) -> new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        ),
        (userCodec, mapperService) -> new KNN910Codec(userCodec),
        KNN910Codec::new
    ),

    V_9_2_0(
        "KNN920Codec",
        new Lucene92Codec(),
        new KNN920PerFieldKnnVectorsFormat(Optional.empty()),
        (delegate) -> new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        ),
        (userCodec, mapperService) -> KNN920Codec.builder()
            .delegate(userCodec)
            .knnVectorsFormat(new KNN920PerFieldKnnVectorsFormat(Optional.ofNullable(mapperService)))
            .build(),
        KNN920Codec::new
    ),

    V_9_4_0(
        "KNN940Codec",
        new Lucene94Codec(),
        new KNN940PerFieldKnnVectorsFormat(Optional.empty()),
        (delegate) -> new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        ),
        (userCodec, mapperService) -> KNN940Codec.builder()
            .delegate(userCodec)
            .knnVectorsFormat(new KNN940PerFieldKnnVectorsFormat(Optional.ofNullable(mapperService)))
            .build(),
        KNN940Codec::new
    );

    private static final KNNCodecVersion CURRENT = V_9_4_0;

    private final String codecName;
    private final Codec defaultCodecDelegate;
    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;
    private final Function<Codec, KNNFormatFacade> knnFormatFacadeSupplier;
    private final BiFunction<Codec, MapperService, Codec> knnCodecSupplier;
    private final Supplier<Codec> defaultKnnCodecSupplier;

    public static final KNNCodecVersion current() {
        return CURRENT;
    }
}
