/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN950Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.junit.Ignore;
import org.opensearch.knn.index.codec.KNNCodecTestCase;

import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_5_0;

public class KNN950CodecTests extends KNNCodecTestCase {

    @SneakyThrows
    @Ignore
    public void testMultiFieldsKnnIndex() {
        testMultiFieldsKnnIndex(KNN950Codec.builder().delegate(V_9_5_0.getDefaultCodecDelegate()).build());
    }

    @SneakyThrows
    @Ignore
    public void testBuildFromModelTemplate() {
        testBuildFromModelTemplate((KNN950Codec.builder().delegate(V_9_5_0.getDefaultCodecDelegate()).build()));
    }

    // Ensure that the codec is able to return the correct per field knn vectors format for codec
    public void testCodecSetsCustomPerFieldKnnVectorsFormat() {
        final Codec codec = new KNN950Codec();
        assertTrue(codec.knnVectorsFormat() instanceof KNN950PerFieldKnnVectorsFormat);
    }
}
