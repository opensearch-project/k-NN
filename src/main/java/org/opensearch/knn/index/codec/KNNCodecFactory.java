/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene91.Lucene91Codec;
import org.opensearch.knn.index.codec.KNN91Codec.KNN91Codec;

import java.lang.reflect.Constructor;
import java.util.Map;

/**
 * Factory abstraction for KNN codes
 */
public class KNNCodecFactory {

    private static Map<KNNCodecVersion, Class> CODEC_BY_VERSION = ImmutableMap.of(KNNCodecVersion.KNN91, KNN91Codec.class);

    private static KNNCodecVersion LATEST_KNN_CODEC_VERSION = KNNCodecVersion.KNN91;

    public static Codec createKNNCodec(final Codec userCodec) {
        return getCodec(LATEST_KNN_CODEC_VERSION, userCodec);
    }

    public static Codec createKNNCodec(final KNNCodecVersion knnCodecVersion, final Codec userCodec) {
        return getCodec(knnCodecVersion, userCodec);
    }

    private static Codec getCodec(final KNNCodecVersion knnCodecVersion, final Codec userCodec) {
        try {
            Constructor<?> constructor = CODEC_BY_VERSION.getOrDefault(knnCodecVersion, CODEC_BY_VERSION.get(LATEST_KNN_CODEC_VERSION))
                .getConstructor(Codec.class);
            return (Codec) constructor.newInstance(userCodec);
        } catch (Exception ex) {
            throw new RuntimeException("Cannot create instance of KNN codec", ex);
        }
    }

    public static class CodecDelegateFactory {

        public static Codec createKNN91DefaultDelegate() {
            return new Lucene91Codec();
        }
    }

    enum KNNCodecVersion {
        KNN91
    }
}
