/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN910Codec;

import org.opensearch.knn.index.codec.KNNCodecTestCase;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

public class KNN910CodecTests extends KNNCodecTestCase {

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(new KNN910Codec());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate(new KNN910Codec());
    }
}
