/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN940Codec.KNN940Codec;
import org.opensearch.knn.index.codec.KNN940Codec.KNN940PerFieldKnnVectorsFormat;

import java.util.Optional;

/**
 * Factory abstraction for KNN codec
 */
@AllArgsConstructor
public class KNNCodecFactory {

    private final MapperService mapperService;

    public Codec createKNNCodec(final Codec userCodec) {
        var codec = KNN940Codec.builder()
            .delegate(userCodec)
            .knnVectorsFormat(new KNN940PerFieldKnnVectorsFormat(Optional.of(mapperService)))
            .build();
        return codec;
    }

    /**
     * Factory abstraction for codec delegate
     */
    public static class CodecDelegateFactory {

        public static Codec createKNNDefaultDelegate(KNNCodecVersion KNNCodecVersion) {
            return KNNCodecVersion.getDefaultCodecDelegate();
        }
    }
}
