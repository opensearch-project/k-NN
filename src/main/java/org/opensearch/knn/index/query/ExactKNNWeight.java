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

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.index.query.ExactSearcher.ExactSearcherContext;

import java.io.IOException;

/**
 * Weight implementation for exact k-NN search using brute-force distance calculations.
 */
public class ExactKNNWeight extends Weight {

    @Getter
    private final ExactKNNQuery exactKNNQuery;
    private static ExactSearcher DEFAULT_EXACT_SEARCHER = new ExactSearcher(null);
    private final float boost;
    private final ExactSearcher exactSearcher;

    public ExactKNNWeight(ExactKNNQuery query, float boost) {
        super(query);
        this.exactKNNQuery = query;
        this.boost = boost;
        this.exactSearcher = DEFAULT_EXACT_SEARCHER;
    }

    public static void initialize(ExactSearcher exactSearcher) {
        ExactKNNWeight.DEFAULT_EXACT_SEARCHER = exactSearcher;
    }

    @Override
    public Explanation explain(LeafReaderContext leafReaderContext, int doc) throws IOException {
        return explain(leafReaderContext, doc, 0);
    }

    public Explanation explain(LeafReaderContext leafReaderContext, int doc, float score) {
        exactKNNQuery.setExplain(true);
        String vectorString = exactKNNQuery.getVectorDataType() == VectorDataType.FLOAT
            ? java.util.Arrays.toString(((ExactKNNFloatQuery) exactKNNQuery).getQueryVector())
            : java.util.Arrays.toString(((ExactKNNByteQuery) exactKNNQuery).getByteQueryVector());
        String description = String.format(
            "exact k-NN search on field [%s] with vector [%s] and space type [%s]",
            exactKNNQuery.getField(),
            vectorString,
            exactKNNQuery.getSpaceType()
        );
        return Explanation.match(score * boost, description);
    }

    @Override
    public ScorerSupplier scorerSupplier(LeafReaderContext leafReaderContext) throws IOException {
        return new ScorerSupplier() {

            @Override
            public Scorer get(long leadCost) throws IOException {
                ExactSearcherContext exactContext = createExactSearcherContext(leafReaderContext);
                KNNIterator knnIterator = exactSearcher.createIterator(leafReaderContext, exactContext);
                if (knnIterator == null) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNExactLazyScorer(knnIterator, boost);
            }

            @Override
            public long cost() {
                return leafReaderContext.reader().maxDoc();
            }
        };
    }

    private ExactSearcherContext createExactSearcherContext(LeafReaderContext leafReaderContext) throws IOException {
        DocIdSetIterator matchedDocsIterator = exactKNNQuery.getParentFilter() == null
            ? DocIdSetIterator.all(leafReaderContext.reader().maxDoc())
            : null;

        switch (exactKNNQuery.getVectorDataType()) {
            case BINARY:
                return ExactSearcher.ExactSearcherContext.builder()
                    .field(exactKNNQuery.getField())
                    .byteQueryVector(((ExactKNNByteQuery) exactKNNQuery).getByteQueryVector())
                    .matchedDocsIterator(matchedDocsIterator)
                    .parentsFilter(exactKNNQuery.getParentFilter())
                    .exactKNNSpaceType(exactKNNQuery.getSpaceType())
                    .build();
            default:
                return ExactSearcher.ExactSearcherContext.builder()
                    .field(exactKNNQuery.getField())
                    .floatQueryVector(((ExactKNNFloatQuery) exactKNNQuery).getQueryVector())
                    .matchedDocsIterator(matchedDocsIterator)
                    .parentsFilter(exactKNNQuery.getParentFilter())
                    .exactKNNSpaceType(exactKNNQuery.getSpaceType())
                    .build();
        }
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
        return true;
    }
}
