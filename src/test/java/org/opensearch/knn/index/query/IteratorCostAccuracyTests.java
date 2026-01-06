/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.index.Term;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

/**
 * Tests to verify the accuracy of cost() for different iterator types.
 * This helps determine which iterator types can reliably use cost() to estimate cardinality
 * for exact search decisions in KNNWeight.
 */
public class IteratorCostAccuracyTests extends KNNTestCase {

    private static final int K = 10;
    private static final int TOTAL_DOCS = 10000;

    // Test cardinalities: low (5), medium (10), high (1000)
    private static final int LOW_CARDINALITY = 5;
    private static final int MEDIUM_CARDINALITY = 10;
    private static final int HIGH_CARDINALITY = 1000;

    private Directory directory;
    private DirectoryReader reader;
    private IndexSearcher searcher;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = new ByteBuffersDirectory();
        IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig());

        for (int i = 0; i < TOTAL_DOCS; i++) {
            Document doc = new Document();
            doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
            // Field for TermQuery tests - distribute values to control cardinality
            doc.add(new StringField("category_low", i < LOW_CARDINALITY ? "match" : "no_match", Field.Store.NO));
            doc.add(new StringField("category_medium", i < MEDIUM_CARDINALITY ? "match" : "no_match", Field.Store.NO));
            doc.add(new StringField("category_high", i < HIGH_CARDINALITY ? "match" : "no_match", Field.Store.NO));
            // Field for PointRangeQuery tests
            doc.add(new IntPoint("value", i));
            writer.addDocument(doc);
        }
        writer.commit();
        writer.close();

        reader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(reader);
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        super.tearDown();
    }

    // ==================== TermQuery Tests ====================

    public void testTermQueryCost_LowCardinality() throws IOException {
        verifyCostAccuracy(new TermQuery(new Term("category_low", "match")), LOW_CARDINALITY, "TermQuery-Low");
    }

    public void testTermQueryCost_MediumCardinality() throws IOException {
        verifyCostAccuracy(new TermQuery(new Term("category_medium", "match")), MEDIUM_CARDINALITY, "TermQuery-Medium");
    }

    public void testTermQueryCost_HighCardinality() throws IOException {
        verifyCostAccuracy(new TermQuery(new Term("category_high", "match")), HIGH_CARDINALITY, "TermQuery-High");
    }

    // ==================== BooleanQuery (MUST/AND) Tests ====================

    public void testBooleanMustQueryCost_LowCardinality() throws IOException {
        Query query = new BooleanQuery.Builder().add(new TermQuery(new Term("category_low", "match")), BooleanClause.Occur.MUST)
            .add(new TermQuery(new Term("category_high", "match")), BooleanClause.Occur.MUST)
            .build();
        // AND of low and high should return low (intersection)
        verifyCostAccuracy(query, LOW_CARDINALITY, "BooleanMUST-Low");
    }

    public void testBooleanMustQueryCost_MediumCardinality() throws IOException {
        Query query = new BooleanQuery.Builder().add(new TermQuery(new Term("category_medium", "match")), BooleanClause.Occur.MUST)
            .add(new TermQuery(new Term("category_high", "match")), BooleanClause.Occur.MUST)
            .build();
        verifyCostAccuracy(query, MEDIUM_CARDINALITY, "BooleanMUST-Medium");
    }

    // ==================== BooleanQuery (SHOULD/OR) Tests ====================

    public void testBooleanShouldQueryCost_LowCardinality() throws IOException {
        // OR of two non-overlapping low cardinality sets
        Query query = new BooleanQuery.Builder().add(new TermQuery(new Term("category_low", "match")), BooleanClause.Occur.SHOULD).build();
        verifyCostAccuracy(query, LOW_CARDINALITY, "BooleanSHOULD-Low");
    }

    public void testBooleanShouldQueryCost_HighCardinality() throws IOException {
        // OR of low and medium - should be union
        Query query = new BooleanQuery.Builder().add(new TermQuery(new Term("category_low", "match")), BooleanClause.Occur.SHOULD)
            .add(new TermQuery(new Term("category_medium", "match")), BooleanClause.Occur.SHOULD)
            .build();
        // Union of [0,5) and [0,10) = [0,10) = 10 docs
        verifyCostAccuracy(query, MEDIUM_CARDINALITY, "BooleanSHOULD-High");
    }

    // ==================== PointRangeQuery Tests ====================

    public void testPointRangeQueryCost_LowCardinality() throws IOException {
        Query query = IntPoint.newRangeQuery("value", 0, LOW_CARDINALITY - 1);
        verifyCostAccuracy(query, LOW_CARDINALITY, "PointRange-Low");
    }

    public void testPointRangeQueryCost_MediumCardinality() throws IOException {
        Query query = IntPoint.newRangeQuery("value", 0, MEDIUM_CARDINALITY - 1);
        verifyCostAccuracy(query, MEDIUM_CARDINALITY, "PointRange-Medium");
    }

    public void testPointRangeQueryCost_HighCardinality() throws IOException {
        Query query = IntPoint.newRangeQuery("value", 0, HIGH_CARDINALITY - 1);
        verifyCostAccuracy(query, HIGH_CARDINALITY, "PointRange-High");
    }

    // ==================== BitSetIterator Tests ====================

    public void testBitSetIteratorCost_LowCardinality() {
        FixedBitSet bitSet = new FixedBitSet(TOTAL_DOCS);
        for (int i = 0; i < LOW_CARDINALITY; i++) {
            bitSet.set(i);
        }
        BitSetIterator iterator = new BitSetIterator(bitSet, LOW_CARDINALITY);
        verifyIteratorCostAccuracy(iterator, LOW_CARDINALITY, "BitSetIterator-Low");
    }

    public void testBitSetIteratorCost_MediumCardinality() {
        FixedBitSet bitSet = new FixedBitSet(TOTAL_DOCS);
        for (int i = 0; i < MEDIUM_CARDINALITY; i++) {
            bitSet.set(i);
        }
        BitSetIterator iterator = new BitSetIterator(bitSet, MEDIUM_CARDINALITY);
        verifyIteratorCostAccuracy(iterator, MEDIUM_CARDINALITY, "BitSetIterator-Medium");
    }

    public void testBitSetIteratorCost_HighCardinality() {
        FixedBitSet bitSet = new FixedBitSet(TOTAL_DOCS);
        for (int i = 0; i < HIGH_CARDINALITY; i++) {
            bitSet.set(i);
        }
        BitSetIterator iterator = new BitSetIterator(bitSet, HIGH_CARDINALITY);
        verifyIteratorCostAccuracy(iterator, HIGH_CARDINALITY, "BitSetIterator-High");
    }

    // ==================== Mixed Query Tests ====================

    public void testMixedQueryCost_TermAndPointRange() throws IOException {
        // AND of term and point range
        Query query = new BooleanQuery.Builder().add(new TermQuery(new Term("category_high", "match")), BooleanClause.Occur.MUST)
            .add(IntPoint.newRangeQuery("value", 0, MEDIUM_CARDINALITY - 1), BooleanClause.Occur.MUST)
            .build();
        // Intersection of [0,1000) and [0,10) = [0,10) = 10 docs
        verifyCostAccuracy(query, MEDIUM_CARDINALITY, "Mixed-TermAndPoint");
    }

    public void testMixedQueryCost_PointRangeAndTerm_LowCardinality() throws IOException {
        // PointRange (low) AND Term (high) - PointRange is lead iterator
        Query query = new BooleanQuery.Builder().add(IntPoint.newRangeQuery("value", 0, LOW_CARDINALITY - 1), BooleanClause.Occur.MUST)
            .add(new TermQuery(new Term("category_high", "match")), BooleanClause.Occur.MUST)
            .build();
        verifyCostAccuracy(query, LOW_CARDINALITY, "Mixed-PointAndTerm-Low");
    }

    public void testMixedQueryCost_TermAndPointRange_TermLeads() throws IOException {
        // Term (low) AND PointRange (high) - Term is lead iterator
        Query query = new BooleanQuery.Builder().add(new TermQuery(new Term("category_low", "match")), BooleanClause.Occur.MUST)
            .add(IntPoint.newRangeQuery("value", 0, HIGH_CARDINALITY - 1), BooleanClause.Occur.MUST)
            .build();
        verifyCostAccuracy(query, LOW_CARDINALITY, "Mixed-TermLeads-Low");
    }

    // ==================== Helper Methods ====================

    private void verifyCostAccuracy(Query query, int expectedCardinality, String testName) throws IOException {
        for (LeafReaderContext leaf : reader.leaves()) {
            Weight weight = query.createWeight(searcher, org.apache.lucene.search.ScoreMode.COMPLETE_NO_SCORES, 1.0f);
            Scorer scorer = weight.scorer(leaf);
            if (scorer == null) {
                continue;
            }

            DocIdSetIterator iterator = scorer.iterator();
            long cost = iterator.cost();
            int actualCount = countDocs(iterator);

            double accuracy = (double) cost / actualCount;
            boolean isExact = cost == actualCount;
            boolean isOverestimate = cost >= actualCount;

            logger.info(
                "{}: cost={}, actual={}, accuracy={}, exact={}, overestimate={}",
                testName,
                cost,
                actualCount,
                String.format("%.2f", accuracy),
                isExact,
                isOverestimate
            );

            // Verify cost is at least as large as actual (should never underestimate)
            assertTrue(testName + ": cost should not underestimate", cost >= actualCount);
        }
    }

    private void verifyIteratorCostAccuracy(DocIdSetIterator iterator, int expectedCardinality, String testName) {
        long cost = iterator.cost();
        int actualCount = countDocs(iterator);

        double accuracy = (double) cost / actualCount;
        boolean isExact = cost == actualCount;

        logger.info("{}: cost={}, actual={}, accuracy={}, exact={}", testName, cost, actualCount, String.format("%.2f", accuracy), isExact);

        assertEquals(testName + ": BitSetIterator cost should be exact", expectedCardinality, cost);
        assertEquals(testName + ": actual count should match expected", expectedCardinality, actualCount);
    }

    private int countDocs(DocIdSetIterator iterator) {
        int count = 0;
        try {
            while (iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                count++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return count;
    }
}
