/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LateInteractionFloatValuesSource;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.common.QueryUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * Two-phase query that reranks the results of an inner query using late-interaction (MaxSim)
 * similarity computed natively over Lucene {@link org.apache.lucene.document.LateInteractionField}
 * binary doc-values.
 *
 * <p>Phase 1: run {@code innerQuery} (typically a kNN search over a single-vector FDE field) to get
 * the candidate set. Phase 2: for each candidate, read its multi-vectors from {@code field} and
 * compute {@code SUM_MAX_SIM} against the query multi-vectors via Lucene's
 * {@link LateInteractionFloatValuesSource}, then keep the top {@code k}.
 *
 * <p>This is the native, doc-values-backed alternative to the {@code _source}-based
 * {@code lateInteractionScore} Painless function; it produces identical scores.
 */
@Log4j2
public class LateInteractionRescoreQuery extends Query {

    private final Query innerQuery;
    private final String field;
    private final int k;
    private final float[][] queryVector;
    private final VectorSimilarityFunction vectorSimilarityFunction;

    /**
     * @param innerQuery               phase-1 query producing the candidate set
     * @param field                    late-interaction field name (binary doc-values)
     * @param k                        number of results to keep after rescoring
     * @param queryVector              query multi-vectors (per-token)
     * @param vectorSimilarityFunction per-vector similarity used inside MaxSim (from the field's space type)
     */
    public LateInteractionRescoreQuery(
        Query innerQuery,
        String field,
        int k,
        float[][] queryVector,
        VectorSimilarityFunction vectorSimilarityFunction
    ) {
        this.innerQuery = innerQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.vectorSimilarityFunction = vectorSimilarityFunction;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        final Query rewrittenInnerQuery = searcher.rewrite(innerQuery);
        final Weight weight = searcher.createWeight(rewrittenInnerQuery, ScoreMode.COMPLETE_NO_SCORES, boost);
        final TopDocs[] perLeafResults = doRescore(searcher, weight);
        final TopDocs topK = TopDocs.merge(k, perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }
        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), topK).createWeight(searcher, scoreMode, boost);
    }

    private TopDocs[] doRescore(final IndexSearcher indexSearcher, final Weight weight) throws IOException {
        final List<LeafReaderContext> leaves = indexSearcher.getIndexReader().leaves();
        final List<Callable<TopDocs>> tasks = new ArrayList<>(leaves.size());
        for (LeafReaderContext leaf : leaves) {
            tasks.add(() -> searchLeaf(weight, leaf));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks).toArray(TopDocs[]::new);
    }

    private TopDocs searchLeaf(final Weight weight, final LeafReaderContext leaf) throws IOException {
        final Scorer scorer = weight.scorer(leaf);
        if (scorer == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        // A DoubleValues over this leaf that computes SUM_MAX_SIM(queryVector, docMultiVectors) per doc.
        final LateInteractionFloatValuesSource valuesSource = new LateInteractionFloatValuesSource(
            field,
            queryVector,
            vectorSimilarityFunction
        );
        final DoubleValues maxSimValues = valuesSource.getValues(leaf, null);

        final DocIdSetIterator iterator = scorer.iterator();
        final List<ScoreDoc> scoreDocs = new ArrayList<>();
        int docId = iterator.nextDoc();
        while (docId != DocIdSetIterator.NO_MORE_DOCS) {
            final float score;
            if (maxSimValues.advanceExact(docId)) {
                score = (float) maxSimValues.doubleValue();
            } else {
                // Candidate has no value for the late-interaction field; drop it to the bottom.
                score = Float.NEGATIVE_INFINITY;
            }
            scoreDocs.add(new ScoreDoc(docId + leaf.docBase, score));
            docId = iterator.nextDoc();
        }

        scoreDocs.sort(Comparator.comparingDouble((ScoreDoc sd) -> sd.score).reversed());
        final int topN = Math.min(k, scoreDocs.size());
        final ScoreDoc[] top = scoreDocs.subList(0, topN).toArray(new ScoreDoc[0]);
        final TopDocs result = new TopDocs(
            new org.apache.lucene.search.TotalHits(top.length, org.apache.lucene.search.TotalHits.Relation.EQUAL_TO),
            top
        );
        return result;
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(String f) {
        return getClass().getSimpleName() + "{field=" + field + ", k=" + k + ", queryTokens=" + queryVector.length + "}";
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        LateInteractionRescoreQuery other = (LateInteractionRescoreQuery) obj;
        return k == other.k
            && Objects.equals(innerQuery, other.innerQuery)
            && Objects.equals(field, other.field)
            && vectorSimilarityFunction == other.vectorSimilarityFunction
            && Arrays.deepEquals(queryVector, other.queryVector);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), innerQuery, field, k, vectorSimilarityFunction, Arrays.deepHashCode(queryVector));
    }
}
