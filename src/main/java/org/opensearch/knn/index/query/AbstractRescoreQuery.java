/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/** Shared second-pass rescoring for top-k and radial vector queries. */
public abstract class AbstractRescoreQuery extends org.apache.lucene.search.Query {
    protected final QueryUtils queryUtils;

    protected AbstractRescoreQuery(final QueryUtils queryUtils) {
        this.queryUtils = queryUtils;
    }

    public QueryUtils getQueryUtils() {
        return queryUtils;
    }

    protected final List<PerLeafResult> rescore(
        final IndexSearcher searcher,
        final List<LeafReaderContext> leaves,
        final List<PerLeafResult> perLeafResults,
        final int firstPassK,
        final boolean shardLevelRescoringDisabled,
        final Supplier<ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder> contextBuilderSupplier,
        final BitSetProducer parentsFilter,
        final LeafExactSearcher exactSearcher
    ) throws IOException {
        if (leaves.size() != perLeafResults.size()) {
            throw new IllegalArgumentException("Leaf contexts and results must have the same size");
        }
        if (shardLevelRescoringDisabled == false) {
            ResultUtil.reduceToTopK(perLeafResults, firstPassK);
        }

        final List<Callable<PerLeafResult>> tasks = new ArrayList<>(leaves.size());
        for (int i = 0; i < leaves.size(); i++) {
            final LeafReaderContext leaf = leaves.get(i);
            final PerLeafResult firstPassResult = perLeafResults.get(i);
            tasks.add(() -> rescoreLeaf(leaf, firstPassResult, contextBuilderSupplier, parentsFilter, exactSearcher));
        }
        return searcher.getTaskExecutor().invokeAll(tasks);
    }

    private PerLeafResult rescoreLeaf(
        final LeafReaderContext leaf,
        final PerLeafResult firstPassResult,
        final Supplier<ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder> contextBuilderSupplier,
        final BitSetProducer parentsFilter,
        final LeafExactSearcher exactSearcher
    ) throws IOException {
        if (firstPassResult.getResult().scoreDocs.length == 0) {
            return firstPassResult;
        }

        final DocIdSetIterator matchedDocs;
        if (parentsFilter == null) {
            matchedDocs = new TopDocsDISI(firstPassResult.getResult());
        } else {
            final Set<Integer> docIds = Arrays.stream(firstPassResult.getResult().scoreDocs)
                .map(scoreDoc -> scoreDoc.doc)
                .collect(Collectors.toSet());
            matchedDocs = queryUtils.getAllSiblings(leaf, docIds, parentsFilter, firstPassResult.getFilterBits());
        }

        final ExactSearcher.ExactSearcherContext context = contextBuilderSupplier.get()
            .matchedDocsIterator(matchedDocs)
            .numberOfMatchedDocs(matchedDocs.cost())
            .useQuantizedVectorsForSearch(false)
            .parentsFilter(parentsFilter)
            .build();
        return new PerLeafResult(
            firstPassResult.getFilterBits(),
            firstPassResult.getFilterBitsCardinality(),
            exactSearcher.search(leaf, context),
            PerLeafResult.SearchMode.EXACT_SEARCH
        );
    }

    @FunctionalInterface
    protected interface LeafExactSearcher {
        TopDocs search(LeafReaderContext leaf, ExactSearcher.ExactSearcherContext context) throws IOException;
    }
}
