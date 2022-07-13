/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.codecs.lucene92.Lucene92Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.util.CodecBuilder;

import java.util.Map;

/**
 * Factory abstraction for KNN codec
 */
public class KNNCodecFactory {

    private final Map<KNNCodecVersion, CodecBuilder> codecByVersion;

    private static final KNNCodecVersion LATEST_KNN_CODEC_VERSION = KNNCodecVersion.KNN920;

    public KNNCodecFactory(MapperService mapperService) {
        codecByVersion = ImmutableMap.of(
            KNNCodecVersion.KNN910,
            new CodecBuilder.KNN91CodecBuilder(mapperService),
            KNNCodecVersion.KNN920,
            new CodecBuilder.KNN92CodecBuilder(mapperService)
        );
    }

    public Codec createKNNCodec(final Codec userCodec) {
        return getCodec(LATEST_KNN_CODEC_VERSION, userCodec);
    }

    public Codec createKNNCodec(final KNNCodecVersion knnCodecVersion, final Codec userCodec) {
        return getCodec(knnCodecVersion, userCodec);
    }

    private Codec getCodec(final KNNCodecVersion knnCodecVersion, final Codec userCodec) {
        try {
            final CodecBuilder codecBuilder = codecByVersion.getOrDefault(knnCodecVersion, codecByVersion.get(LATEST_KNN_CODEC_VERSION));
            return codecBuilder.userCodec(userCodec).build();
        } catch (Exception ex) {
            throw new RuntimeException("Cannot create instance of KNN codec", ex);
        }
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

    /**
     * Collection of supported KNN codec versions
     */
    enum KNNCodecVersion {
        KNN910,
        KNN920
    }
}
