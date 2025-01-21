/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene912.Lucene912Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;
import org.opensearch.knn.index.codec.KNNFormatFacade;
import org.opensearch.knn.index.codec.WrapperCodecForKNNPlugin;
import org.opensearch.knn.index.codec.jvector.JVectorCompoundFormat;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * Example for a specific codec version that extends {@link WrapperCodecForKNNPlugin} and provides codec specific delegates
 * This makes allows for all the constructors and resources to be encapsulated in a single class without going back and forth between providers
 */
public class KNN9120WraperCodec extends WrapperCodecForKNNPlugin {
    private static final String CODEC_NAME = "KNN9120Codec";

    public KNN9120WraperCodec(String name, MapperService mapperService) {
        super(name,
                new Lucene912Codec(),
                CODEC_NAME,
                mapperService,
                new KNN9120PerFieldKnnVectorsFormat(Optional.empty()),
                (delegate) -> new KNNFormatFacade(
                        new KNN80DocValuesFormat(delegate.docValuesFormat()),
                        new JVectorCompoundFormat(delegate.compoundFormat())
                ));
    }
}
