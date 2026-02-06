/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.CustomCodec;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.codec.KNNCodecVersion;

import java.util.Optional;
import java.util.function.Function;

public class KNN1030CodecTests extends KNNCodecTestCase {

    @SneakyThrows
    public void testMultiFieldsKnnIndex() {
        testMultiFieldsKnnIndex(KNN1030Codec.builder().delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodec() {
        testMultiFieldsKnnIndex(KNN1030Codec.builder().delegate(new CustomCodec()).build());
    }

    @SneakyThrows
    public void testBuildFromModelTemplate() {
        testBuildFromModelTemplate(KNN1030Codec.builder().delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    // Ensure that the codec is able to return the correct per field knn vectors format for codec
    public void testCodecSetsCustomPerFieldKnnVectorsFormat() {
        final Codec codec = new KNN1030Codec();
        assertTrue(codec.knnVectorsFormat() instanceof KNN9120PerFieldKnnVectorsFormat);
    }

    // IMPORTANT: When this Codec is moved to a backwards Codec, this test needs to be removed, because it attempts to
    // write with a read-only codec, which will fail
    @SneakyThrows
    public void testKnnVectorIndex() {
        Function<MapperService, PerFieldKnnVectorsFormat> perFieldKnnVectorsFormatProvider = (
            mapperService) -> new KNN9120PerFieldKnnVectorsFormat(Optional.of(mapperService));

        Function<PerFieldKnnVectorsFormat, Codec> knnCodecProvider = (knnVectorFormat) -> KNN1030Codec.builder()
            .delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE)
            .knnVectorsFormat(knnVectorFormat)
            .build();

        testKnnVectorIndex(knnCodecProvider, perFieldKnnVectorsFormatProvider);
    }
}
