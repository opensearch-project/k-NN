/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN950Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNNCodecTestCase;

import java.util.Optional;
import java.util.function.Function;

import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_5_0;

public class KNN950CodecTests extends KNNCodecTestCase {

    @SneakyThrows
    public void testMultiFieldsKnnIndex() {
        testMultiFieldsKnnIndex(KNN950Codec.builder().delegate(V_9_5_0.getDefaultCodecDelegate()).build());
    }

    @SneakyThrows
    public void testBuildFromModelTemplate() {
        testBuildFromModelTemplate((KNN950Codec.builder().delegate(V_9_5_0.getDefaultCodecDelegate()).build()));
    }

    @SneakyThrows
    public void testKnnVectorIndex() {
        Function<MapperService, PerFieldKnnVectorsFormat> perFieldKnnVectorsFormatProvider = (
            mapperService) -> new KNN950PerFieldKnnVectorsFormat(Optional.of(mapperService));

        Function<PerFieldKnnVectorsFormat, Codec> knnCodecProvider = (knnVectorFormat) -> KNN950Codec.builder()
            .delegate(V_9_5_0.getDefaultCodecDelegate())
            .knnVectorsFormat(knnVectorFormat)
            .build();

        testKnnVectorIndex(knnCodecProvider, perFieldKnnVectorsFormatProvider);
    }
}
