/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.LateInteractionField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.LateInteractionFloatValuesSource;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;

import java.util.ArrayList;
import java.util.List;

/**
 * Correctness tests for the native late-interaction (MaxSim) scoring used by
 * {@link LateInteractionRescoreQuery}, via Lucene's {@link LateInteractionFloatValuesSource} configured
 * with our {@link RawMaxSimilarity}.
 *
 * <p>The scores are asserted against an independent raw-MaxSim oracle (sum of the best raw per-token
 * similarity, mapped once through the space's score transform). This deliberately does <b>not</b> assert
 * parity with Lucene's default {@code SUM_MAX_SIM}, which sums the <em>scaled</em> per-token
 * {@link VectorSimilarityFunction#compare} and can rank out of MaxSim order for {@code l2} and negative-dot
 * {@code innerproduct} (k-NN #3612). The ranking cases below use the exact counterexamples from #3612.
 */
public class LateInteractionScoreParityTests extends KNNTestCase {

    private static final String FIELD = "tokens";

    @SneakyThrows
    public void testNativeMaxSimMatchesRawOracle_innerProduct() {
        assertMatchesRawOracle(SpaceType.INNER_PRODUCT);
    }

    @SneakyThrows
    public void testNativeMaxSimMatchesRawOracle_cosine() {
        assertMatchesRawOracle(SpaceType.COSINESIMIL);
    }

    @SneakyThrows
    public void testNativeMaxSimMatchesRawOracle_l2() {
        assertMatchesRawOracle(SpaceType.L2);
    }

    /**
     * #3612 counterexample: with query tokens (1,0,0) and (0,1,0), doc B (0.55,0.55,0.6285) has a smaller
     * sum of per-query-token minimum squared distances than doc A (1,0,0), so B is the MaxSim winner under
     * l2. Lucene's scaled SUM_MAX_SIM ranks A first; RawMaxSimilarity must rank B first.
     */
    @SneakyThrows
    public void testNativeRanking_l2_ranksByMaxSim() {
        final float[][] query = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } };
        final float[][] docA = { { 1.0f, 0.0f, 0.0f } };
        final float[][] docB = { { 0.55f, 0.55f, 0.6285f } };
        final List<Float> scores = nativeScores(SpaceType.L2, query, docA, docB);
        assertTrue("doc B (lower sum of min squared distances) must outrank doc A under l2", scores.get(1) > scores.get(0));
    }

    /**
     * #3612 counterexample: with query tokens (1,0) and (0,1), doc A (0.9,-0.6) has MaxSim 0.3 and doc B
     * (0.2,0.2) has MaxSim 0.4, so B wins. Lucene's scaled SUM_MAX_SIM ranks A first because of the negative
     * dot product; RawMaxSimilarity must rank B first.
     */
    @SneakyThrows
    public void testNativeRanking_innerProduct_negativeDot_ranksByMaxSim() {
        final float[][] query = { { 1.0f, 0.0f }, { 0.0f, 1.0f } };
        final float[][] docA = { { 0.9f, -0.6f } };
        final float[][] docB = { { 0.2f, 0.2f } };
        final List<Float> scores = nativeScores(SpaceType.INNER_PRODUCT, query, docA, docB);
        assertTrue("doc B (higher MaxSim) must outrank doc A under innerproduct with a negative dot", scores.get(1) > scores.get(0));
    }

    @SneakyThrows
    private void assertMatchesRawOracle(SpaceType spaceType) {
        final float[][] query = { { 1.0f, 0.0f, 0.5f, -0.2f }, { 0.3f, 0.9f, -0.1f, 0.4f }, { -0.5f, 0.2f, 0.8f, 0.1f } };

        final List<float[][]> docs = new ArrayList<>();
        docs.add(new float[][] { { 0.9f, 0.1f, 0.4f, -0.1f }, { 0.2f, 0.8f, 0.0f, 0.3f } });
        docs.add(new float[][] { { -0.4f, 0.3f, 0.7f, 0.2f }, { 0.1f, 0.1f, 0.1f, 0.1f }, { 0.6f, -0.2f, 0.5f, 0.0f } });
        docs.add(new float[][] { { 0.0f, 0.0f, 1.0f, 0.0f } });

        final List<Float> nativeScores = nativeScores(spaceType, query, docs.toArray(new float[0][][]));
        assertEquals(docs.size(), nativeScores.size());
        for (int i = 0; i < docs.size(); i++) {
            final float expected = rawMaxSimOracle(query, docs.get(i), spaceType);
            assertEquals(
                "native vs raw-MaxSim oracle mismatch for space=" + spaceType.getValue() + " doc=" + i,
                expected,
                nativeScores.get(i),
                1e-4f
            );
        }
    }

    /** Runs the native value source (same path as the rescore query) over the given docs, in index order. */
    private List<Float> nativeScores(SpaceType spaceType, float[][] query, float[][]... docs) throws Exception {
        final VectorSimilarityFunction simFn = spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
        final RawMaxSimilarity rawMaxSimilarity = new RawMaxSimilarity(spaceType);
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig())) {
                for (float[][] doc : docs) {
                    Document d = new Document();
                    d.add(new LateInteractionField(FIELD, doc));
                    writer.addDocument(d);
                }
            }
            try (IndexReader reader = DirectoryReader.open(dir)) {
                final LateInteractionFloatValuesSource valuesSource = new LateInteractionFloatValuesSource(
                    FIELD,
                    query,
                    simFn,
                    rawMaxSimilarity
                );
                final List<Float> scores = new ArrayList<>();
                for (LeafReaderContext leaf : reader.leaves()) {
                    final DoubleValues values = valuesSource.getValues(leaf, null);
                    for (int i = 0; i < leaf.reader().maxDoc(); i++) {
                        assertTrue("expected a value for doc " + i, values.advanceExact(i));
                        scores.add((float) values.doubleValue());
                    }
                }
                return scores;
            }
        }
    }

    /**
     * Independent oracle: sum of the best raw per-token similarity, mapped once through the space transform.
     * Mirrors {@code KNNPainlessScriptUtils#lateInteractionScore} (#3613) and {@link RawMaxSimilarity}.
     */
    private float rawMaxSimOracle(float[][] query, float[][] doc, SpaceType spaceType) {
        final boolean isL2 = spaceType == SpaceType.L2;
        double rawSum = 0.0;
        for (float[] q : query) {
            float best = isL2 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            for (float[] d : doc) {
                switch (spaceType) {
                    case L2:
                        best = Math.min(best, VectorUtil.squareDistance(q, d));
                        break;
                    case COSINESIMIL:
                        best = Math.max(best, VectorUtil.cosine(q, d));
                        break;
                    default:
                        best = Math.max(best, VectorUtil.dotProduct(q, d));
                }
            }
            rawSum += best;
        }
        final double mean = rawSum / query.length;
        switch (spaceType) {
            case L2:
                return (float) (query.length / (1 + mean));
            case COSINESIMIL:
                return (float) (query.length * (1 + mean) / 2);
            default:
                return (float) (query.length * (mean < 0 ? 1 / (1 - mean) : mean + 1));
        }
    }
}
