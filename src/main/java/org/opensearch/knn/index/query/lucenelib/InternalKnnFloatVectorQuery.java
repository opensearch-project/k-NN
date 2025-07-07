/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;

/**
 * InternalKnnVectorQuery for float vector
 */
@Log4j2
public class InternalKnnFloatVectorQuery extends KnnFloatVectorQuery {
    @Getter
    protected String exactSearchSpaceType;
    private static ModelDao modelDao;
    private static ExactSearcher exactSearcher;

    public InternalKnnFloatVectorQuery(String field, float[] target, int k, Query filter, String exactSearchSpaceType) {
        super(field, target, k, filter);
        this.exactSearchSpaceType = exactSearchSpaceType;
    }

    public static void initialize(ModelDao modelDao) {
        InternalKnnFloatVectorQuery.modelDao = modelDao;
        InternalKnnFloatVectorQuery.exactSearcher = new ExactSearcher(modelDao);
    }

    public TopDocs searchLeaf(LeafReaderContext context, int k, Query filter) throws IOException {
        if (exactSearcher == null) {
            throw new IllegalStateException("ExactSearcher not initialized. Call initialize() first.");
        }

        ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
            .parentsFilter(null)
            .k(k)
            // setting to true, so that if quantization details are present we want to do search on the quantized
            // vectors as this flow is used in first pass of search.
            .useQuantizedVectorsForSearch(true)
            .field(field)
            .floatQueryVector(target)
            // setting to false since memory optimized search only enabled for hnsw
            .isMemoryOptimizedSearchEnabled(false)
            .exactSearchSpaceType(exactSearchSpaceType)
            .isLuceneExactSearch(true)
            .build();
        return exactSearcher.searchLeaf(context, exactSearcherContext);
    }
}
