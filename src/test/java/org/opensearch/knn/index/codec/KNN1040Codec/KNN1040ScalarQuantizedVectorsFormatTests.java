/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.engine.KNNEngine;

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
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertEquals("KNN1040ScalarQuantizedVectorsFormat", format.getName());
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_returnsPrefetchableScorer() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);

        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            SegmentReadState readState = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());

            try (FlatVectorsReader reader = format.fieldsReader(readState)) {
                RandomVectorScorer scorer = reader.getRandomVectorScorer(
                    KNN1040ScalarQuantizedTestUtils.FIELD_NAME,
                    KNN1040ScalarQuantizedTestUtils.randomVector(KNN1040ScalarQuantizedTestUtils.DIMENSION, random())
                );

                assertNotNull("RandomVectorScorer should not be null", scorer);
                assertTrue(
                    "Scorer should be PrefetchableRandomVectorScorer (backed by KNN1040ScalarQuantizedVectorScorer), "
                        + "but was: "
                        + scorer.getClass().getSimpleName(),
                    scorer instanceof PrefetchableRandomVectorScorer
                );
                assertEquals("maxOrd should match number of vectors", KNN1040ScalarQuantizedTestUtils.NUM_VECTORS, scorer.maxOrd());
            }
        }
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
