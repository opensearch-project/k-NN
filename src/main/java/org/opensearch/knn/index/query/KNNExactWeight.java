/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.DocIdSetIterator;

import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.index.query.ExactSearcher.ExactSearcherContext;

import java.io.IOException;

/**
 * Weight implementation for exact k-NN search using brute-force distance calculations.
 */
public class KNNExactWeight extends Weight {

    @Getter
    private final KNNExactQuery knnExactQuery;
    private static ModelDao modelDao;
    private static ExactSearcher DEFAULT_EXACT_SEARCHER;
    private final float boost;
    private final ExactSearcher exactSearcher;

    public KNNExactWeight(KNNExactQuery query, float boost) {
        super(query);
        this.knnExactQuery = query;
        this.boost = boost;
        this.exactSearcher = DEFAULT_EXACT_SEARCHER;
    }

    public static void initialize(ModelDao modelDao) {
        initialize(modelDao, new ExactSearcher(modelDao));
    }

    public static void initialize(ModelDao modelDao, ExactSearcher exactSearcher) {
        KNNExactWeight.modelDao = modelDao;
        KNNExactWeight.DEFAULT_EXACT_SEARCHER = exactSearcher;
    }

    @Override
    public Explanation explain(LeafReaderContext context, int doc) throws IOException {
        return explain(context, doc, 0);
    }

    public Explanation explain(LeafReaderContext context, int doc, float score) {
        knnExactQuery.setExplain(true);
        String vectorString = knnExactQuery.getQueryVector() != null
            ? java.util.Arrays.toString(knnExactQuery.getQueryVector())
            : java.util.Arrays.toString(knnExactQuery.getByteQueryVector());
        String description = String.format("exact k-NN search on field [%s] with vector [%s]", knnExactQuery.getField(), vectorString);
        return Explanation.match(score * boost, description);
    }

    @Override
    public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
        return new ScorerSupplier() {

            @Override
            public Scorer get(long leadCost) throws IOException {
                ExactSearcherContext exactContext = createExactSearcherContext(context);
                KNNIterator knnIterator = exactSearcher.createIterator(context, exactContext);
                if (knnIterator == null) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNLazyScorer(knnIterator, boost);
            }

            @Override
            public long cost() {
                return context.reader().maxDoc();
            }
        };
    }

    private ExactSearcherContext createExactSearcherContext(LeafReaderContext context) throws IOException {
        String userDefinedSpaceType = knnExactQuery.getSpaceType() != null ? knnExactQuery.getSpaceType() : null;
        return ExactSearcher.ExactSearcherContext.builder()
            .field(knnExactQuery.getField())
            .parentsFilter(knnExactQuery.getParentFilter())
            .floatQueryVector(knnExactQuery.getQueryVector())
            .byteQueryVector(knnExactQuery.getByteQueryVector())
            .matchedDocsIterator(DocIdSetIterator.all(context.reader().maxDoc()))
            .exactKNNSpaceType(userDefinedSpaceType)
            .build();
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
        return true;
    }
}
