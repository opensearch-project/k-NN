/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN920Codec;

import org.junit.Ignore;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_2_0;

public class KNN920CodecTests extends KNNCodecTestCase {

    @Ignore
    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(KNN920Codec.builder().delegate(V_9_2_0.getDefaultCodecDelegate()).build());
    }

    @Ignore
    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate((KNN920Codec.builder().delegate(V_9_2_0.getDefaultCodecDelegate()).build()));
    }
}
