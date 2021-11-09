/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN87Codec;

import org.opensearch.knn.index.codec.KNNCodecTestCase;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

public class KNN87CodecTests extends KNNCodecTestCase {

    public void testFooter() throws Exception {
        testFooter(new KNN87Codec());
    }

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(new KNN87Codec());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate(new KNN87Codec());
    }
}
