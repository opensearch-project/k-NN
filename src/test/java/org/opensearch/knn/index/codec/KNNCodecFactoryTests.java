/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene91.Lucene91Codec;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN910Codec.KNN910Codec;

public class KNNCodecFactoryTests extends KNNTestCase {

    public void testKNN91DefaultDelegate() {
        Codec knn91DefaultDelegate = KNNCodecFactory.CodecDelegateFactory.createKNN91DefaultDelegate();
        assertNotNull(knn91DefaultDelegate);
        assertTrue(knn91DefaultDelegate instanceof Lucene91Codec);
    }

    public void testKNN91DefaultCodec() {
        Lucene91Codec lucene91CodecDelegate = new Lucene91Codec();
        Codec knnCodec = KNNCodecFactory.createKNNCodec(lucene91CodecDelegate);
        assertNotNull(knnCodec);
        assertTrue(knnCodec instanceof KNN910Codec);
    }

    public void testKNN91CodecByVersion() {
        Lucene91Codec lucene91CodecDelegate = new Lucene91Codec();
        Codec knnCodec = KNNCodecFactory.createKNNCodec(KNNCodecFactory.KNNCodecVersion.KNN910, lucene91CodecDelegate);
        assertNotNull(knnCodec);
        assertTrue(knnCodec instanceof KNN910Codec);
    }
}
