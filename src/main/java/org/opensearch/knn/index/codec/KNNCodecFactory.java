/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.backward_codecs.lucene92.Lucene92Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN920Codec.KNN920Codec;
import org.opensearch.knn.index.codec.KNN920Codec.KNN920PerFieldKnnVectorsFormat;

import java.util.Optional;

/**
 * Factory abstraction for KNN codec
 */
@AllArgsConstructor
public class KNNCodecFactory {

    private final MapperService mapperService;

    public Codec createKNNCodec(final Codec userCodec) {
        var codec = KNN920Codec.builder()
            .delegate(userCodec)
            .knnVectorsFormat(new KNN920PerFieldKnnVectorsFormat(Optional.of(mapperService)))
            .build();
        return codec;
    }

    /**
     * Factory abstraction for codec delegate
     */
    public static class CodecDelegateFactory {

        public static Codec createKNN91DefaultDelegate() {
            return new Lucene91Codec();
        }

        public static Codec createKNN92DefaultDelegate() {
            return new Lucene92Codec();
        }
    }
}
