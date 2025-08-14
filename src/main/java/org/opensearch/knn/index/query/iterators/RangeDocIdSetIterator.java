/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

public class RangeDocIdSetIterator extends DocIdSetIterator {
    private final int minDoc;
    private final int maxDoc;
    private final DocIdSetIterator _innerIter;
    private int doc = -1;

    public RangeDocIdSetIterator(DocIdSetIterator inner, int minDoc, int maxDoc) {
        if (minDoc < 0 || maxDoc <= minDoc) {
            throw new IllegalArgumentException("Invalid range: [" + minDoc + "," + maxDoc + ")");
        }
        _innerIter = inner;
        this.minDoc = minDoc;
        this.maxDoc = maxDoc;
    }

    public int docID() {
        return doc;
    }

    public long cost() {
        return _innerIter.cost();
    }

    @Override
    public int nextDoc() throws IOException {
        if (doc < minDoc) {
            return advance(minDoc);
        }
        return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
        target = Math.max(target, minDoc);
        doc = _innerIter.advance(target);
        if (doc >= maxDoc) {
            doc = NO_MORE_DOCS;
        }
        return doc;
    }
}
