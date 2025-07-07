/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.index.query.KNNScorer;
import org.opensearch.knn.index.query.lucenelib.InternalKnnByteVectorQuery;
import org.opensearch.knn.index.query.lucenelib.InternalKnnFloatVectorQuery;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
        Query rewrittenQuery = luceneQuery.rewrite(searcher);
        if (luceneQuery instanceof InternalKnnFloatVectorQuery || luceneQuery instanceof InternalKnnByteVectorQuery) {
            return new Weight(this) {
                @Override
                public Explanation explain(LeafReaderContext context, int doc) throws IOException {
                    float score = 0;
                    try {
                        final Scorer scorer = scorer(context);
                        assert scorer != null;
                        int resDoc = scorer.iterator().advance(doc);
                        if (resDoc == doc) {
                            score = scorer.score();
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    List<Explanation> details = new ArrayList<>();
                    if (luceneQuery instanceof InternalKnnFloatVectorQuery) {
                        InternalKnnFloatVectorQuery floatQuery = (InternalKnnFloatVectorQuery) luceneQuery;
                        details.add(
                            Explanation.match(score, "Executed KNN exact search with space type: " + floatQuery.getExactSearchSpaceType())
                        );
                    } else {
                        InternalKnnByteVectorQuery byteQuery = (InternalKnnByteVectorQuery) luceneQuery;
                        details.add(
                            Explanation.match(score, "Executed KNN exact search with space type: " + byteQuery.getExactSearchSpaceType())
                        );
                    }
                    return Explanation.match(score, "The type of search executed was KNN exact search", details);
                }

                @Override
                public boolean isCacheable(LeafReaderContext ctx) {
                    return false;
                }

                @Override
                public ScorerSupplier scorerSupplier(LeafReaderContext context) {
                    return new ScorerSupplier() {
                        long cost = -1L;

                        @Override
                        public Scorer get(long leadCost) throws IOException {
                            final TopDocs topDocs;
                            if (luceneQuery instanceof InternalKnnFloatVectorQuery) {
                                InternalKnnFloatVectorQuery floatQuery = (InternalKnnFloatVectorQuery) luceneQuery;
                                topDocs = floatQuery.searchLeaf(context, floatQuery.getK(), floatQuery.getFilter());
                            } else {
                                InternalKnnByteVectorQuery byteQuery = (InternalKnnByteVectorQuery) luceneQuery;
                                topDocs = byteQuery.searchLeaf(context, byteQuery.getK(), byteQuery.getFilter());
                            }
                            cost = topDocs.scoreDocs.length;
                            if (cost == 0) {
                                return KNNScorer.emptyScorer();
                            }
                            return new KNNScorer(topDocs, boost);
                        }

                        @Override
                        public long cost() {
                            if (luceneQuery instanceof InternalKnnFloatVectorQuery) {
                                InternalKnnFloatVectorQuery floatQuery = (InternalKnnFloatVectorQuery) luceneQuery;
                                return cost == -1L ? floatQuery.getK() : cost;
                            } else {
                                InternalKnnByteVectorQuery byteQuery = (InternalKnnByteVectorQuery) luceneQuery;
                                return cost == -1L ? byteQuery.getK() : cost;
                            }
                        }
                    };
                }
            };
        }
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
