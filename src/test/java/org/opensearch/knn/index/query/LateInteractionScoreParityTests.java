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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNPainlessScriptUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Correctness parity test: the native late-interaction scoring (Lucene {@code SUM_MAX_SIM} over
 * {@link LateInteractionField} doc-values, as used by {@link LateInteractionRescoreQuery}) must
 * produce the same MaxSim score as the existing {@code _source}-based
 * {@link KNNPainlessScriptUtils#lateInteractionScore} Painless function.
 */
public class LateInteractionScoreParityTests extends KNNTestCase {

    private static final String FIELD = "tokens";

    @SneakyThrows
    public void testNativeMaxSimMatchesPainless_innerProduct() {
        assertParity(SpaceType.INNER_PRODUCT);
    }

    @SneakyThrows
    public void testNativeMaxSimMatchesPainless_cosine() {
        assertParity(SpaceType.COSINESIMIL);
    }

    @SneakyThrows
    public void testNativeMaxSimMatchesPainless_l2() {
        assertParity(SpaceType.L2);
    }

    @SneakyThrows
    private void assertParity(SpaceType spaceType) {
        final float[][] query = { { 1.0f, 0.0f, 0.5f, -0.2f }, { 0.3f, 0.9f, -0.1f, 0.4f }, { -0.5f, 0.2f, 0.8f, 0.1f } };

        final List<float[][]> docs = new ArrayList<>();
        docs.add(new float[][] { { 0.9f, 0.1f, 0.4f, -0.1f }, { 0.2f, 0.8f, 0.0f, 0.3f } });
        docs.add(new float[][] { { -0.4f, 0.3f, 0.7f, 0.2f }, { 0.1f, 0.1f, 0.1f, 0.1f }, { 0.6f, -0.2f, 0.5f, 0.0f } });
        docs.add(new float[][] { { 0.0f, 0.0f, 1.0f, 0.0f } });

        final VectorSimilarityFunction simFn = spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();

        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig())) {
                for (float[][] doc : docs) {
                    Document d = new Document();
                    d.add(new LateInteractionField(FIELD, doc));
                    writer.addDocument(d);
                }
            }

            try (IndexReader reader = DirectoryReader.open(dir)) {
                // Native scores, in index order, via Lucene's value source (same path as the rescore query).
                final LateInteractionFloatValuesSource valuesSource = new LateInteractionFloatValuesSource(FIELD, query, simFn);
                final List<Float> nativeScores = new ArrayList<>();
                for (LeafReaderContext leaf : reader.leaves()) {
                    final DoubleValues values = valuesSource.getValues(leaf, null);
                    for (int i = 0; i < leaf.reader().maxDoc(); i++) {
                        assertTrue("expected a value for doc " + i, values.advanceExact(i));
                        nativeScores.add((float) values.doubleValue());
                    }
                }

                // Painless scores for the same query/doc vectors, in the same order.
                assertEquals(docs.size(), nativeScores.size());
                for (int i = 0; i < docs.size(); i++) {
                    final float painless = painlessScore(query, docs.get(i), spaceType);
                    assertEquals(
                        "native vs painless mismatch for space=" + spaceType.getValue() + " doc=" + i,
                        painless,
                        nativeScores.get(i),
                        1e-4f
                    );
                }
            }
        }
    }

    /** Invokes the production Painless function with the given vectors. */
    private float painlessScore(float[][] query, float[][] doc, SpaceType spaceType) {
        final List<List<Number>> q = toNumberLists(query);
        final List<List<Number>> d = toNumberLists(doc);
        final Map<String, Object> source = new HashMap<>();
        source.put(FIELD, d);
        return KNNPainlessScriptUtils.lateInteractionScore(q, FIELD, source, spaceType.getValue());
    }

    private List<List<Number>> toNumberLists(float[][] vectors) {
        final List<List<Number>> out = new ArrayList<>();
        for (float[] v : vectors) {
            final List<Number> row = new ArrayList<>();
            for (float f : v) {
                row.add(f);
            }
            out.add(row);
        }
        return out;
    }
}
