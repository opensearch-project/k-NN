/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.lucenelib.ExpandNestedDocsQuery;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.search.profile.query.QueryProfiler;
import org.opensearch.knn.index.query.common.QueryUtils;

import java.io.IOException;

/**
 * LuceneEngineKnnVectorQuery is a wrapper around a vector queries for the Lucene engine.
 * This enables us to defer rewrites until weight creation to optimize repeated execution
 * of Lucene based k-NN queries.
 */
@AllArgsConstructor
@EqualsAndHashCode(callSuper = false)
@Log4j2
public class LuceneEngineKnnVectorQuery extends Query {
    @Getter
    private final Query luceneQuery;
    private final int luceneK; // Number of results requested from Lucene engine (may be > k for better recall)
    private final int k; // Final number of results to return to user

    /*
      Prevents repeated rewrites of the query for the Lucene engine.
    */
    @Override
    public Query rewrite(IndexSearcher indexSearcher) {
        return this;
    }

    /*
       Rewrites the query just before weight creation.
     */
    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        QueryProfiler profiler = KNNProfileUtil.getProfiler(searcher);
        if (profiler != null) {
            profiler.getQueryBreakdown(luceneQuery);
        }
        Query rewrittenQuery = luceneQuery.rewrite(searcher);
        Query docAndScoreQuery = reduceToTopK(rewrittenQuery, searcher, scoreMode, boost);
        final Weight weight = docAndScoreQuery.createWeight(searcher, scoreMode, boost);
        if (profiler != null) {
            profiler.pollLastElement();
        }
        return weight;
    }

    private Query reduceToTopK(Query query, IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {

        // Skip reducing to top-k in two cases:
        // 1. When luceneK equals k (no reduction needed)
        // 2. When query is ExpandNestedDocsQuery (reducing would exclude required child documents)
        if (luceneK == k || query instanceof ExpandNestedDocsQuery) {
            return query;
        }

        Weight weight = query.createWeight(searcher, scoreMode, boost);
        TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE);

        for (LeafReaderContext context : searcher.getIndexReader().leaves()) {
            Scorer scorer = weight.scorer(context);
            if (scorer != null) {
                DocIdSetIterator iterator = scorer.iterator();
                int doc;
                while ((doc = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
                    collector.collect(doc + context.docBase, scorer.score());
                }
            }
        }
        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), collector.topDocs());
    }

    @Override
    public String toString(String s) {
        return "LuceneEngineKnnVectorQuery[luceneK=" + luceneK + ", k=" + k + ", query=" + luceneQuery.toString() + "]";
    }

    @Override
    public void visit(QueryVisitor queryVisitor) {
        queryVisitor.visitLeaf(this);
    }
}
