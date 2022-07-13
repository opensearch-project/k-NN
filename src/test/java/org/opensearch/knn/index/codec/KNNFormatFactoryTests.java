/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;

public class KNNFormatFactoryTests extends KNNTestCase {

    public void testKNN91Format() {
        final Codec lucene91CodecDelegate = KNNCodecFactory.CodecDelegateFactory.createKNN91DefaultDelegate();
        MapperService mapperService = mock(MapperService.class);
        KNNCodecFactory knnCodecFactory = new KNNCodecFactory(mapperService);
        final Codec knnCodec = knnCodecFactory.createKNNCodec(lucene91CodecDelegate);
        KNNFormatFacade knnFormatFacade = KNNFormatFactory.createKNN910Format(knnCodec);

        assertNotNull(knnFormatFacade);
        assertNotNull(knnFormatFacade.compoundFormat());
        assertNotNull(knnFormatFacade.docValuesFormat());
    }

    public void testKNN92Format() {
        final Codec lucene92CodecDelegate = KNNCodecFactory.CodecDelegateFactory.createKNN92DefaultDelegate();
        MapperService mapperService = mock(MapperService.class);
        KNNCodecFactory knnCodecFactory = new KNNCodecFactory(mapperService);
        final Codec knnCodec = knnCodecFactory.createKNNCodec(lucene92CodecDelegate);
        KNNFormatFacade knnFormatFacade = KNNFormatFactory.createKNN920Format(knnCodec);

        assertNotNull(knnFormatFacade);
        assertNotNull(knnFormatFacade.compoundFormat());
        assertNotNull(knnFormatFacade.docValuesFormat());
    }
}
