/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Calculates query weights and builds query scorers to implement exact search for Lucene queries
 */
@Log4j2
public class LuceneKNNWeight extends Weight {
    private static ModelDao modelDao;
    private static ExactSearcher DEFAULT_EXACT_SEARCHER;

    private final float boost;
    private final LuceneEngineKnnVectorQuery luceneQuery;
    private final ExactSearcher exactSearcher;

    public LuceneKNNWeight(LuceneEngineKnnVectorQuery query, float boost) {
        super(query);
        this.boost = boost;
        this.luceneQuery = query;
        this.exactSearcher = DEFAULT_EXACT_SEARCHER;

    }

    public static void initialize(ModelDao modelDao) {
        initialize(modelDao, new ExactSearcher(modelDao));
    }

    static void initialize(ModelDao modelDao, ExactSearcher exactSearcher) {
        LuceneKNNWeight.modelDao = modelDao;
        LuceneKNNWeight.DEFAULT_EXACT_SEARCHER = exactSearcher;
    }

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
        details.add(Explanation.match(score, "Executed KNN exact search with space type: " + luceneQuery.getExactSearchSpaceType()));
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
                if (luceneQuery.getLuceneQuery() instanceof KnnFloatVectorQuery) {
                    KnnFloatVectorQuery floatQuery = (KnnFloatVectorQuery) luceneQuery.getLuceneQuery();
                    topDocs = searchLeaf(context, floatQuery.getK());
                } else {
                    KnnByteVectorQuery byteQuery = (KnnByteVectorQuery) luceneQuery.getLuceneQuery();
                    topDocs = searchLeaf(context, byteQuery.getK());
                }
                cost = topDocs.scoreDocs.length;
                if (cost == 0) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNScorer(topDocs, boost);
            }

            @Override
            public long cost() {
                if (luceneQuery.getLuceneQuery() instanceof KnnFloatVectorQuery) {
                    KnnFloatVectorQuery floatQuery = (KnnFloatVectorQuery) luceneQuery.getLuceneQuery();
                    return cost == -1L ? floatQuery.getK() : cost;
                } else {
                    KnnByteVectorQuery byteQuery = (KnnByteVectorQuery) luceneQuery.getLuceneQuery();
                    return cost == -1L ? byteQuery.getK() : cost;
                }
            }
        };
    }

    public TopDocs searchLeaf(LeafReaderContext context, int k) throws IOException {
        if (exactSearcher == null) {
            throw new IllegalStateException("exactSearcher not initialized. Call initialize() first.");
        }
        ExactSearcher.ExactSearcherContext exactSearcherContext = null;
        if (luceneQuery.getLuceneQuery() instanceof KnnFloatVectorQuery) {
            KnnFloatVectorQuery floatQuery = (KnnFloatVectorQuery) luceneQuery.getLuceneQuery();
            exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                .parentsFilter(null)
                .k(k)
                // setting to true, so that if quantization details are present we want to do search on the quantized
                // vectors as this flow is used in first pass of search.
                .useQuantizedVectorsForSearch(true)
                .field(floatQuery.getField())
                .floatQueryVector(floatQuery.getTargetCopy())
                // setting to false since memory optimized search only enabled for hnsw
                .isMemoryOptimizedSearchEnabled(false)
                .exactSearchSpaceType(luceneQuery.getExactSearchSpaceType())
                .isLuceneExactSearch(true)
                .build();
        } else if (luceneQuery.getLuceneQuery() instanceof KnnByteVectorQuery) {
            KnnByteVectorQuery byteQuery = (KnnByteVectorQuery) luceneQuery.getLuceneQuery();
            float[] floatVector = convertByteToFloatArray(byteQuery.getTargetCopy());
            exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                .parentsFilter(null)
                .k(k)
                // setting to true, so that if quantization details are present we want to do search on the quantized
                // vectors as this flow is used in first pass of search.
                .useQuantizedVectorsForSearch(true)
                .field(byteQuery.getField())
                .floatQueryVector(floatVector)
                .byteQueryVector(byteQuery.getTargetCopy())
                // setting to false since memory optimized search only enabled for hnsw
                .isMemoryOptimizedSearchEnabled(false)
                .exactSearchSpaceType(luceneQuery.getExactSearchSpaceType())
                .isLuceneExactSearch(true)
                .build();
        }
        return exactSearcher.searchLeaf(context, exactSearcherContext);
    }

    private float[] convertByteToFloatArray(byte[] byteArray) {
        float[] floatArray = new float[byteArray.length];
        for (int i = 0; i < byteArray.length; i++) {
            floatArray[i] = byteArray[i];
        }
        return floatArray;
    }
}
