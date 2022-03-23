/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN87Codec;

import org.opensearch.knn.index.codec.KNNCodecTestCase;

public class KNN87CodecTests extends KNNCodecTestCase {

    public void testWriteByOldCodec() throws Exception {
        testWriteByOldCodec(new KNN87Codec());
    }
}
