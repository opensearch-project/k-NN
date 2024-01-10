/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.backward_codecs.lucene92.Lucene92Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.backward_codecs.lucene94.Lucene94Codec;
import org.apache.lucene.backward_codecs.lucene95.Lucene95Codec;
import org.opensearch.knn.KNNTestCase;

import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_1_0;
import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_2_0;
import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_4_0;
import static org.opensearch.knn.index.codec.KNNCodecVersion.V_9_5_0;

public class KNNCodecFactoryTests extends KNNTestCase {

    public void testKNN910Codec() {
        assertDelegateForVersion(V_9_1_0, Lucene91Codec.class);
        assertNull(V_9_1_0.getPerFieldKnnVectorsFormat());
        assertNotNull(V_9_1_0.getKnnFormatFacadeSupplier().apply(V_9_1_0.getDefaultCodecDelegate()));
    }

    public void testKNN920Codec() {
        assertDelegateForVersion(V_9_2_0, Lucene92Codec.class);
        assertNotNull(V_9_2_0.getPerFieldKnnVectorsFormat());
        assertNotNull(V_9_2_0.getKnnFormatFacadeSupplier().apply(V_9_2_0.getDefaultCodecDelegate()));
    }

    public void testKNN940Codec() {
        assertDelegateForVersion(V_9_4_0, Lucene94Codec.class);
        assertNotNull(V_9_4_0.getPerFieldKnnVectorsFormat());
        assertNotNull(V_9_4_0.getKnnFormatFacadeSupplier().apply(V_9_4_0.getDefaultCodecDelegate()));
    }

    public void testKNN950Codec() {
        assertDelegateForVersion(V_9_5_0, Lucene95Codec.class);
        assertNotNull(V_9_5_0.getPerFieldKnnVectorsFormat());
        assertNotNull(V_9_5_0.getKnnFormatFacadeSupplier().apply(V_9_5_0.getDefaultCodecDelegate()));
    }

    private void assertDelegateForVersion(final KNNCodecVersion codecVersion, final Class expectedCodecClass) {
        final Codec defaultDelegate = codecVersion.getDefaultCodecDelegate();
        assertNotNull(defaultDelegate);
        assertTrue(defaultDelegate.getClass().isAssignableFrom(expectedCodecClass));
    }
}
