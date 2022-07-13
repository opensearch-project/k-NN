/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.codecs.lucene92.Lucene92Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN910Codec.KNN910Codec;
import org.opensearch.knn.index.codec.KNN920Codec.KNN920Codec;

import static org.mockito.Mockito.mock;

public class KNNCodecFactoryTests extends KNNTestCase {

    public void testKNN91DefaultDelegate() {
        Codec knn91DefaultDelegate = KNNCodecFactory.CodecDelegateFactory.createKNN91DefaultDelegate();
        assertNotNull(knn91DefaultDelegate);
        assertTrue(knn91DefaultDelegate instanceof Lucene91Codec);
    }

    public void testKNN92DefaultDelegate() {
        Codec knn92DefaultDelegate = KNNCodecFactory.CodecDelegateFactory.createKNN92DefaultDelegate();
        assertNotNull(knn92DefaultDelegate);
        assertTrue(knn92DefaultDelegate instanceof Lucene92Codec);
    }

    public void testKNNDefaultCodec() {
        Lucene91Codec lucene91CodecDelegate = new Lucene91Codec();
        MapperService mapperService = mock(MapperService.class);
        KNNCodecFactory knnCodecFactory = new KNNCodecFactory(mapperService);
        Codec knnCodec = knnCodecFactory.createKNNCodec(lucene91CodecDelegate);
        assertNotNull(knnCodec);
        assertTrue(knnCodec instanceof KNN920Codec);
    }

    public void testKNN91CodecByVersion() {
        Lucene91Codec lucene91CodecDelegate = new Lucene91Codec();
        MapperService mapperService = mock(MapperService.class);
        KNNCodecFactory knnCodecFactory = new KNNCodecFactory(mapperService);
        Codec knnCodec = knnCodecFactory.createKNNCodec(KNNCodecFactory.KNNCodecVersion.KNN910, lucene91CodecDelegate);
        assertNotNull(knnCodec);
        assertTrue(knnCodec instanceof KNN910Codec);
    }

    public void testKNN92CodecByVersion() {
        Lucene92Codec lucene92CodecDelegate = new Lucene92Codec();
        MapperService mapperService = mock(MapperService.class);
        KNNCodecFactory knnCodecFactory = new KNNCodecFactory(mapperService);
        Codec knnCodec = knnCodecFactory.createKNNCodec(KNNCodecFactory.KNNCodecVersion.KNN920, lucene92CodecDelegate);
        assertNotNull(knnCodec);
        assertTrue(knnCodec instanceof KNN920Codec);
    }
}
