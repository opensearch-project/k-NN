/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.index.codec.CodecService;
import org.opensearch.index.codec.CodecServiceConfig;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN1030Codec.KNN1030Codec;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

import java.util.Optional;

/**
 * KNNCodecService to inject the right KNNCodec version
 */
public class KNNCodecService extends CodecService {

    private final MapperService mapperService;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public KNNCodecService(CodecServiceConfig codecServiceConfig, NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        super(
            codecServiceConfig.getMapperService(),
            codecServiceConfig.getIndexSettings(),
            codecServiceConfig.getLogger(),
            codecServiceConfig.getAdditionalCodecs()
        );
        mapperService = codecServiceConfig.getMapperService();
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Return the custom k-NN codec that wraps another codec that a user wants to use for non k-NN related operations
     *
     * @param name of delegate codec.
     * @return Latest KNN Codec built with delegate codec.
     */
    @Override
    public Codec codec(String name) {
        return KNN1030Codec.builder()
            .delegate(super.codec(name))
            .mapperService(mapperService)
            .knnVectorsFormat(new KNN9120PerFieldKnnVectorsFormat(Optional.ofNullable(mapperService), nativeIndexBuildStrategyFactory))
            .build();
    }
}
