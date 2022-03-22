/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.opensearch.index.codec.CodecServiceConfig;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.codec.CodecService;

/**
 * KNNCodecService to inject the right KNNCodec version
 */
public class KNNCodecService extends CodecService {

    public KNNCodecService(CodecServiceConfig codecServiceConfig) {
        super(codecServiceConfig.getMapperService(), codecServiceConfig.getLogger());
    }

    /**
     * Return the custom k-NN codec that wraps another codec that a user wants to use for non k-NN related operations
     *
     * @param name of delegate codec.
     * @return Latest KNN Codec built with delegate codec.
     */
    @Override
    public Codec codec(String name) {
        return KNNCodecFactory.createKNN91Codec(super.codec(name));
    }
}
