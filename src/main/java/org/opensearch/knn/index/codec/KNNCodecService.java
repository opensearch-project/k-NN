/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec;

import org.opensearch.index.codec.CodecServiceConfig;
import org.opensearch.knn.index.codec.KNN87Codec.KNN87Codec;
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
        return new KNN87Codec(super.codec(name));
    }
}
