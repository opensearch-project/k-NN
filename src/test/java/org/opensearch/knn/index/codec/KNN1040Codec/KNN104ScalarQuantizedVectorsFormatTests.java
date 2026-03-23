/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.opensearch.knn.KNNTestCase;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN104ScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testToString_containsEncodingAndScorerInfo() {
        KNN104ScalarQuantizedVectorsFormat format = new KNN104ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        String str = format.toString();
        assertTrue("toString should contain class name", str.contains("KNN104ScalarQuantizedVectorsFormat"));
        assertTrue("toString should contain encoding", str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
        assertTrue("toString should contain scorer", str.contains("ScalarQuantizedVectorScorer"));
    }

    public void testConstructor_allEncodings() {
        for (Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding encoding : Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding
            .values()) {
            KNN104ScalarQuantizedVectorsFormat format = new KNN104ScalarQuantizedVectorsFormat(encoding);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }
}
