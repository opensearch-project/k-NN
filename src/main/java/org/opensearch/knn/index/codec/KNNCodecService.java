/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.opensearch.index.codec.CodecServiceConfig;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.codec.CodecService;
import org.opensearch.index.mapper.MapperService;

/**
 * KNNCodecService to inject the right KNNCodec version
 */
public class KNNCodecService extends CodecService {

    private final MapperService mapperService;

    public KNNCodecService(CodecServiceConfig codecServiceConfig) {
        super(codecServiceConfig.getMapperService(), codecServiceConfig.getIndexSettings(), codecServiceConfig.getLogger());
        mapperService = codecServiceConfig.getMapperService();
    }

    /**
     * Return the custom k-NN codec that wraps another codec that a user wants to use for non k-NN related operations
     *
     * @param name of delegate codec.
     * @return Latest KNN Codec built with delegate codec.
     */
    @Override
    public Codec codec(String name) {
        return KNNCodecVersion.current().getKnnCodecSupplier().apply(super.codec(name), mapperService);
    }
}
