/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.index.codec.CodecServiceConfig;
import org.opensearch.knn.index.codec.KNN87Codec.KNN87Codec;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.codec.CodecService;

/**
 * KNNCodecService to inject the right KNNCodec version
 */
class KNNCodecService extends CodecService {

    KNNCodecService(CodecServiceConfig codecServiceConfig) {
        super(codecServiceConfig.getMapperService(), codecServiceConfig.getLogger());
    }

    /**
     * If the index is of type KNN i.e index.knn = true, We always
     * return the KNN Codec
     *
     * @param name dummy name
     * @return Latest KNN Codec
     */
    @Override
    public Codec codec(String name) {
        Codec codec = Codec.forName(KNN87Codec.KNN_87);
        if (codec == null) {
            throw new IllegalArgumentException("failed to find codec [" + name + "]");
        }
        return codec;
    }
}
