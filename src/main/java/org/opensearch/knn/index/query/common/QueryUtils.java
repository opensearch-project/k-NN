/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.common;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.TopDocsDISI;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.iterators.GroupedNestedDocIdSetIterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * This class contains utility methods that help customize the search results
 */
public class QueryUtils {
    // Private constructor to prevent instantiation
    private QueryUtils() {}

    private static final QueryUtils INSTANCE = new QueryUtils();

    // Public method to get the singleton instance
    public static QueryUtils getInstance() {
        return INSTANCE;
    }

    /**
     * Returns a query that represents the specified TopDocs
     * This is copied from org.apache.lucene.search.AbstractKnnVectorQuery#createRewrittenQuery
     *
     * @param reader the index reader
     * @param topDocs the documents to be returned by the query
     * @return a query representing the given TopDocs
     */
    public Query createDocAndScoreQuery(final IndexReader reader, final TopDocs topDocs) {
        return createDocAndScoreQuery(reader, topDocs, null);
    }

    public Query createDocAndScoreQuery(final IndexReader reader, final TopDocs topDocs, final KNNWeight knnWeight) {
        int len = topDocs.scoreDocs.length;
        Arrays.sort(topDocs.scoreDocs, Comparator.comparingInt(a -> a.doc));
        int[] docs = new int[len];
        float[] scores = new float[len];
        for (int i = 0; i < len; i++) {
            docs[i] = topDocs.scoreDocs[i].doc;
            scores[i] = topDocs.scoreDocs[i].score;
        }
        int[] segmentStarts = findSegmentStarts(reader, docs);
        return new DocAndScoreQuery(docs, scores, segmentStarts, reader.getContext().id(), knnWeight);
    }

    private int[] findSegmentStarts(final IndexReader reader, final int[] docs) {
        int[] starts = new int[reader.leaves().size() + 1];
        starts[starts.length - 1] = docs.length;
        if (starts.length == 2) {
            return starts;
        }
        int resultIndex = 0;
        for (int i = 1; i < starts.length - 1; i++) {
            int upper = reader.leaves().get(i).docBase;
            resultIndex = Arrays.binarySearch(docs, resultIndex, docs.length, upper);
            if (resultIndex < 0) {
                resultIndex = -1 - resultIndex;
            }
            starts[i] = resultIndex;
        }
        return starts;
    }

    /**
     * Performs the search in parallel.
     *
     * @param indexSearcher the index searcher
     * @param leafReaderContexts the leaf reader contexts
     * @param weight the search weight
     * @return a list of maps, each mapping document IDs to their scores
     * @throws IOException
     */
    public List<Map<Integer, Float>> doSearch(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leafReaderContexts,
        final Weight weight
    ) throws IOException {
        List<Callable<Map<Integer, Float>>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, weight));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks);
    }

    private Map<Integer, Float> searchLeaf(final LeafReaderContext ctx, final Weight weight) throws IOException {
        Map<Integer, Float> leafDocScores = new HashMap<>();
        Scorer scorer = weight.scorer(ctx);
        if (scorer == null) {
            return Collections.emptyMap();
        }

        DocIdSetIterator iterator = scorer.iterator();
        iterator.nextDoc();
        while (iterator.docID() != DocIdSetIterator.NO_MORE_DOCS) {
            leafDocScores.put(scorer.docID(), scorer.score());
            iterator.nextDoc();
        }
        return leafDocScores;
    }

    @FunctionalInterface
    public interface LeafExactSearcher {
        TopDocs search(LeafReaderContext leaf, ExactSearcher.ExactSearcherContext context) throws IOException;
    }

    /**
     * Applies the first-pass candidate budget and rescores the survivors against full-precision vectors.
     * Results and leaves must be positionally aligned.
     */
    public List<PerLeafResult> rescore(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leaves,
        final List<PerLeafResult> perLeafResults,
        final int firstPassK,
        final boolean shardLevelRescoringDisabled,
        final Supplier<ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder> contextBuilderSupplier,
        final BitSetProducer parentsFilter,
        final LeafExactSearcher exactSearcher
    ) throws IOException {
        validateLeafResults(leaves, perLeafResults);
        if (shardLevelRescoringDisabled == false) {
            ResultUtil.reduceToTopK(perLeafResults, firstPassK);
        }

        final List<Callable<PerLeafResult>> rescoreTasks = new ArrayList<>(perLeafResults.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            final LeafReaderContext leaf = leaves.get(i);
            final PerLeafResult firstPass = perLeafResults.get(i);
            rescoreTasks.add(() -> {
                if (firstPass.getResult().scoreDocs.length == 0) {
                    return firstPass;
                }

                final DocIdSetIterator matchedDocs;
                if (parentsFilter == null) {
                    matchedDocs = new TopDocsDISI(firstPass.getResult());
                } else {
                    final Set<Integer> docIds = Arrays.stream(firstPass.getResult().scoreDocs)
                        .map(scoreDoc -> scoreDoc.doc)
                        .collect(Collectors.toSet());
                    matchedDocs = getAllSiblings(leaf, docIds, parentsFilter, firstPass.getFilterBits());
                }

                final ExactSearcher.ExactSearcherContext context = contextBuilderSupplier.get()
                    .useQuantizedVectorsForSearch(false)
                    .matchedDocsIterator(matchedDocs)
                    .numberOfMatchedDocs(matchedDocs.cost())
                    .parentsFilter(parentsFilter)
                    .build();

                return new PerLeafResult(
                    firstPass.getFilterBits(),
                    firstPass.getFilterBitsCardinality(),
                    exactSearcher.search(leaf, context),
                    PerLeafResult.SearchMode.EXACT_SEARCH
                );
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    /**
     * Converts leaf-local document IDs to shard-level IDs and merges every result.
     */
    public TopDocs mergeLeafResults(final List<LeafReaderContext> leaves, final List<PerLeafResult> perLeafResults) {
        final int resultCount = perLeafResults.stream().mapToInt(result -> result.getResult().scoreDocs.length).sum();
        return mergeLeafResults(leaves, perLeafResults, resultCount);
    }

    /**
     * Converts leaf-local document IDs to shard-level IDs and merges the best {@code topN} results.
     */
    public TopDocs mergeLeafResults(final List<LeafReaderContext> leaves, final List<PerLeafResult> perLeafResults, final int topN) {
        validateLeafResults(leaves, perLeafResults);
        if (topN == 0) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        final TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            final TopDocs leafTopDocs = perLeafResults.get(i).getResult();
            final ScoreDoc[] shardScoreDocs = new ScoreDoc[leafTopDocs.scoreDocs.length];
            for (int j = 0; j < leafTopDocs.scoreDocs.length; j++) {
                final ScoreDoc leafScoreDoc = leafTopDocs.scoreDocs[j];
                final ScoreDoc shardScoreDoc = new ScoreDoc(leafScoreDoc.doc + leaves.get(i).docBase, leafScoreDoc.score);
                shardScoreDoc.shardIndex = leafScoreDoc.shardIndex;
                shardScoreDocs[j] = shardScoreDoc;
            }
            topDocs[i] = new TopDocs(leafTopDocs.totalHits, shardScoreDocs);
        }
        return TopDocs.merge(topN, topDocs);
    }

    private void validateLeafResults(final List<LeafReaderContext> leaves, final List<PerLeafResult> perLeafResults) {
        if (leaves.size() != perLeafResults.size()) {
            throw new IllegalArgumentException("Leaf contexts and results must have the same size");
        }
    }

    /**
     * For the specified nested field document IDs, retrieves all sibling nested field document IDs.
     *
     * @param leafReaderContext the leaf reader context
     * @param docIds the document IDs of the nested field
     * @param parentsFilter a bitset mapping parent document IDs to their nested field document IDs
     * @return an iterator of document IDs for all filtered sibling nested field documents corresponding to the given document IDs
     * @throws IOException
     */
    public DocIdSetIterator getAllSiblings(
        final LeafReaderContext leafReaderContext,
        final Set<Integer> docIds,
        final BitSetProducer parentsFilter,
        final Bits queryFilter
    ) throws IOException {
        if (docIds.isEmpty()) {
            return DocIdSetIterator.empty();
        }

        BitSet parentBitSet = parentsFilter.getBitSet(leafReaderContext);
        return new GroupedNestedDocIdSetIterator(parentBitSet, docIds, queryFilter);
    }

    /**
     * Converts the specified search weight into a {@link Bits} containing document IDs.
     *
     * @param leafReaderContext the leaf reader context
     * @param filterWeight the search weight
     * @return a {@link Bits} of document IDs derived from the search weight
     * @throws IOException
     */
    public Bits createBits(final LeafReaderContext leafReaderContext, final Weight filterWeight) throws IOException {
        if (filterWeight == null) {
            return new Bits.MatchAllBits(0);
        }

        final Scorer scorer = filterWeight.scorer(leafReaderContext);
        if (scorer == null) {
            return new Bits.MatchNoBits(0);
        }

        final Bits liveDocs = leafReaderContext.reader().getLiveDocs();
        final int maxDoc = leafReaderContext.reader().maxDoc();
        DocIdSetIterator filteredDocIdsIterator = scorer.iterator();
        if (liveDocs == null && filteredDocIdsIterator instanceof BitSetIterator) {
            // If we already have a BitSet and no deletions, reuse the BitSet
            return ((BitSetIterator) filteredDocIdsIterator).getBitSet();
        }
        // Create a new BitSet from matching and live docs
        FilteredDocIdSetIterator filterIterator = new FilteredDocIdSetIterator(filteredDocIdsIterator) {
            @Override
            protected boolean match(int doc) {
                return liveDocs == null || liveDocs.get(doc);
            }
        };
        return BitSet.of(filterIterator, maxDoc);
    }
}
