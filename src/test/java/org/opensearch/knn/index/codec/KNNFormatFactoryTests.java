/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.KNNTestCase;

public class KNNFormatFactoryTests extends KNNTestCase {

    public void testKNN91Format() {
        final Codec lucene91CodecDelegate = KNNCodecFactory.CodecDelegateFactory.createKNN91DefaultDelegate();
        final Codec knnCodec = KNNCodecFactory.createKNNCodec(lucene91CodecDelegate);
        KNNFormatFacade knnFormatFacade = KNNFormatFactory.createKNN910Format(knnCodec);

        assertNotNull(knnFormatFacade);
        assertNotNull(knnFormatFacade.compoundFormat());
        assertNotNull(knnFormatFacade.docValuesFormat());
    }
}
