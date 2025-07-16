/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;

/**
 * LuceneEngineKnnVectorQuery is a wrapper around a vector queries for the Lucene engine.
 * This enables us to defer rewrites until weight creation to optimize repeated execution
 * of Lucene based k-NN queries.
 */
@AllArgsConstructor
@Log4j2
public class LuceneEngineKnnVectorQuery extends Query {
    @Getter
    private final Query luceneQuery;

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
        return rewrittenQuery.createWeight(searcher, scoreMode, boost);
    }

    @Override
    public String toString(String s) {
        return luceneQuery.toString();
    }

    @Override
    public void visit(QueryVisitor queryVisitor) {
        queryVisitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LuceneEngineKnnVectorQuery otherQuery = (LuceneEngineKnnVectorQuery) o;
        return luceneQuery.equals(otherQuery.luceneQuery);
    }

    @Override
    public int hashCode() {
        return luceneQuery.hashCode();
    }
}
