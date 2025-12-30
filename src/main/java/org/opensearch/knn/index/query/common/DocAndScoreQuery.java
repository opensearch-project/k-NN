/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.common;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.KNNScorer;
import org.opensearch.knn.index.query.KNNWeight;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

final class DocAndScoreQuery extends Query {

    private final int k;
    private final int[] docs;
    private final float[] scores;
    private final int[] segmentStarts;
    private final Object contextIdentity;
    private final KNNWeight knnWeight;

    public DocAndScoreQuery(int k, int[] docs, float[] scores, int[] segmentStarts, Object contextIdentity, KNNWeight knnWeight) {
        this.k = k;
        this.docs = docs;
        this.scores = scores;
        this.segmentStarts = segmentStarts;
        this.contextIdentity = contextIdentity;
        this.knnWeight = knnWeight;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        if (searcher.getIndexReader().getContext().id() != contextIdentity) {
            throw new IllegalStateException("This DocAndScore query was created by a different reader");
        }

        return new Weight(this) {
            @Override
            public Explanation explain(LeafReaderContext context, int doc) {
                int found = Arrays.binarySearch(docs, doc + context.docBase);
                if (found < 0) {
                    return Explanation.noMatch("not in top " + k);
                }

                // This is copied from org.apache.lucene.search.DocAndScoreQuery#createWeight
                // for LuceneEngineKnnVectorQuery where knnWeight is null
                if (knnWeight == null) {
                    return Explanation.match(scores[found] * boost, "within top " + docs.length + " docs");
                }

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

                return knnWeight.explain(context, doc, score);
            }

            @Override
            public int count(LeafReaderContext context) {
                return segmentStarts[context.ord + 1] - segmentStarts[context.ord];
            }

            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
                return new ScorerSupplier() {
                    @Override
                    public Scorer get(long leadCost) throws IOException {
                        if (segmentStarts[context.ord] == segmentStarts[context.ord + 1]) {
                            return KNNScorer.emptyScorer();
                        }
                        return new Scorer() {
                            final int lower = segmentStarts[context.ord];
                            final int upper = segmentStarts[context.ord + 1];
                            int upTo = -1;

                            @Override
                            public DocIdSetIterator iterator() {
                                return new DocIdSetIterator() {
                                    @Override
                                    public int docID() {
                                        return docIdNoShadow();
                                    }

                                    @Override
                                    public int nextDoc() {
                                        if (upTo == -1) {
                                            upTo = lower;
                                        } else {
                                            ++upTo;
                                        }
                                        return docIdNoShadow();
                                    }

                                    @Override
                                    public int advance(int target) throws IOException {
                                        return slowAdvance(target);
                                    }

                                    @Override
                                    public long cost() {
                                        return upper - lower;
                                    }
                                };
                            }

                            @Override
                            public float getMaxScore(int docId) {
                                docId += context.docBase;
                                float maxScore = 0;
                                for (int idx = Math.max(0, upTo); idx < upper && docs[idx] <= docId; idx++) {
                                    maxScore = Math.max(maxScore, scores[idx]);
                                }
                                return maxScore * boost;
                            }

                            @Override
                            public float score() {
                                return scores[upTo] * boost;
                            }

                            @Override
                            public int advanceShallow(int docid) {
                                int start = Math.max(upTo, lower);
                                int docidIndex = Arrays.binarySearch(docs, start, upper, docid + context.docBase);
                                if (docidIndex < 0) {
                                    docidIndex = -1 - docidIndex;
                                }
                                if (docidIndex >= upper) {
                                    return NO_MORE_DOCS;
                                }
                                return docs[docidIndex];
                            }

                            /**
                             * move the implementation of docID() into a differently-named method so we can call it
                             * from DocIDSetIterator.docID() even though this class is anonymous
                             *
                             * @return the current docid
                             */
                            private int docIdNoShadow() {
                                if (upTo == -1) {
                                    return -1;
                                }
                                if (upTo >= upper) {
                                    return NO_MORE_DOCS;
                                }
                                return docs[upTo] - context.docBase;
                            }

                            @Override
                            public int docID() {
                                return docIdNoShadow();
                            }
                        };

                    }

                    @Override
                    public long cost() {
                        // Estimate the cost of the scoring operation, if applicable.
                        return docs.length == 0 ? k : docs.length;
                    }
                };
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return true;
            }
        };
    }

    @Override
    public String toString(String field) {
        return "DocAndScore[" + k + "][docs:" + Arrays.toString(docs) + ", scores:" + Arrays.toString(scores) + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        return contextIdentity == ((DocAndScoreQuery) obj).contextIdentity
            && Arrays.equals(docs, ((DocAndScoreQuery) obj).docs)
            && Arrays.equals(scores, ((DocAndScoreQuery) obj).scores);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), contextIdentity, Arrays.hashCode(docs), Arrays.hashCode(scores));
    }
}
