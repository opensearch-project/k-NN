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
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.KNNWeight;
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

/**
 * This class contains utility methods that help customize the search results
 */
public class QueryUtils {
    public static QueryUtils INSTANCE = new QueryUtils();

    /**
     * Returns a query that represents the specified TopDocs
     * This is copied from {@link org.apache.lucene.search.AbstractKnnVectorQuery#createRewrittenQuery}
     *
     * @param reader the index reader
     * @param topDocs the documents to be retured by the query
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
        return new DocAndScoreQuery(len, docs, scores, segmentStarts, reader.getContext().id(), knnWeight);
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
