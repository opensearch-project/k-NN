/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1040ScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testToString_containsEncodingAndScorerInfo() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        String str = format.toString();
        assertTrue("toString should contain class name", str.contains("KNN1040ScalarQuantizedVectorsFormat"));
        assertTrue("toString should contain encoding", str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
        assertTrue("toString should contain scorer", str.contains("ScalarQuantizedVectorScorer"));
    }

    public void testDefaultConstructor_usesSingleBitQueryNibble() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertTrue(format.toString().contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertEquals(BuiltinKNNEngine.getMaxDimensionByEngine(BuiltinKNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertEquals("KNN1040ScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_allEncodings() {
        for (Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding encoding : Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding
            .values()) {
            KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(encoding);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }
}
