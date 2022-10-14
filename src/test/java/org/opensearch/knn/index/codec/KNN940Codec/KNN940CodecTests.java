/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN940Codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;

import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_4_0;

public class KNN940CodecTests extends KNNCodecTestCase {

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(KNN940Codec.builder().delegate(V_9_4_0.getDefaultCodecDelegate()).build());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate((KNN940Codec.builder().delegate(V_9_4_0.getDefaultCodecDelegate()).build()));
    }

    public void testKnnVectorIndex() throws Exception {
        Function<MapperService, PerFieldKnnVectorsFormat> perFieldKnnVectorsFormatProvider = (
            mapperService) -> new KNN940PerFieldKnnVectorsFormat(Optional.of(mapperService));

        Function<PerFieldKnnVectorsFormat, Codec> knnCodecProvider = (knnVectorFormat) -> KNN940Codec.builder()
            .delegate(V_9_4_0.getDefaultCodecDelegate())
            .knnVectorsFormat(knnVectorFormat)
            .build();

        testKnnVectorIndex(knnCodecProvider, perFieldKnnVectorsFormatProvider);
    }
}
