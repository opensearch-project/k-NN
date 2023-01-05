/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN940Codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_4_0;

public class KNN940CodecTests extends KNNCodecTestCase {

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(KNN940Codec.builder().delegate(V_9_4_0.getDefaultCodecDelegate()).build());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate((KNN940Codec.builder().delegate(V_9_4_0.getDefaultCodecDelegate()).build()));
    }

    // Ensure that the codec is able to return the correct per field knn vectors format for codec
    public void testCodecSetsCustomPerFieldKnnVectorsFormat() {
        final Codec codec = KNN940Codec.builder()
            .delegate(V_9_4_0.getDefaultCodecDelegate())
            .knnVectorsFormat(V_9_4_0.getPerFieldKnnVectorsFormat())
            .build();

        assertTrue(codec.knnVectorsFormat() instanceof KNN940PerFieldKnnVectorsFormat);
    }
}
