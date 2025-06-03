/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TopDocs;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

/**
 * A DocIdSetIterator implementation that iterates over documents from TopDocs in sorted order.
 * This class maintains document IDs and their corresponding scores from the search results.
 * The implementation is inspired from
 * <a href="https://github.com/apache/lucene/blob/17a40bd1137837fee924a8ac4b2d4c9c1af16307/lucene/core/src/java/org/apache/lucene/search/SeededKnnVectorQuery.java#L228">SeededKnnVectorQuery.java</a>
 */
public class TopDocsDISI extends DocIdSetIterator {

    private final int[] sortedDocIds;
    private final float[] scores;
    private int idx = -1;

    public TopDocsDISI(TopDocs topDocs) {
        // Sort documents by document ID
        Arrays.sort(topDocs.scoreDocs, Comparator.comparingInt(a -> a.doc));
        sortedDocIds = new int[topDocs.scoreDocs.length];
        scores = new float[topDocs.scoreDocs.length];
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            sortedDocIds[i] = topDocs.scoreDocs[i].doc;
            scores[i] = topDocs.scoreDocs[i].score;
        }
    }

    @Override
    public int docID() {
        if (idx == -1) {
            return idx;
        }
        if (idx >= sortedDocIds.length) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        return sortedDocIds[idx];
    }

    @Override
    public int nextDoc() throws IOException {
        idx += 1;
        return docID();
    }

    @Override
    public int advance(int target) throws IOException {
        return slowAdvance(target);
    }

    @Override
    public long cost() {
        return sortedDocIds.length;
    }

    public float score() {
        if (idx == -1) {
            return idx;
        }
        if (idx >= sortedDocIds.length) {
            return Float.MAX_VALUE;
        }
        return scores[idx];
    }
}
