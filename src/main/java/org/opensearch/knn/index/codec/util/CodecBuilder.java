/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.util;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN910Codec.KNN910Codec;
import org.opensearch.knn.index.codec.KNN920Codec.KNN920Codec;

/**
 * Abstracts builder logic for plugin codecs. For each codec we need to set delegate that is typically
 * Lucene codec implementation and this is made part of the base class. Exact builder implementation may add
 * additional parameters required to build a codec.
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
     * @return instance of codec
     */
    public abstract Codec build();

    /**
     * Implements builder abstraction for KNN91Codec, adds MapperService that may be required to build
     * per field format based on field mapper type
     */
    @AllArgsConstructor
    public static class KNN91CodecBuilder extends CodecBuilder {
        private final MapperService mapperService;

        @Override
        public Codec build() {
            return new KNN910Codec(userCodec);
        }
    }

    /**
     * Implements builder abstraction for KNN91Codec, adds MapperService that may be required to build
     * per field format based on field mapper type
     */
    @AllArgsConstructor
    public static class KNN92CodecBuilder extends CodecBuilder {
        private final MapperService mapperService;

        @Override
        public Codec build() {
            return new KNN920Codec(userCodec);
        }
    }
}
