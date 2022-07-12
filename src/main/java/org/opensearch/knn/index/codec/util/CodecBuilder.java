/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.util;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN910Codec.KNN910Codec;

/**
 * Abstracts builder logic for plugin codecs
 */
public abstract class CodecBuilder {
    Codec userCodec;

    /**
     * Set user defined codec for plugin
     * @param userCodec
     * @return
     */
    public CodecBuilder userCodec(Codec userCodec) {
        this.userCodec = userCodec;
        return this;
    }

    /**
     * Builds instance of codec, implementation is specific for each codec version
     * @return
     */
    public abstract Codec build();

    /**
     * Implements builder abstraction for KNN91Codec
     */
    @AllArgsConstructor
    public static class KNN91CodecBuilder extends CodecBuilder {
        private final MapperService mapperService;

        @Override
        public Codec build() {
            return new KNN910Codec(userCodec);
        }
    }
}
