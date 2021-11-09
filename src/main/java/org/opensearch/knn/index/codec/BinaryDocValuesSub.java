/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.MergeState;

import java.io.IOException;

/**
 * Place holder class for the docIds and the respective
 * binary values.
 */
public class BinaryDocValuesSub extends DocIDMerger.Sub {

    private final BinaryDocValues values;

    public BinaryDocValues getValues() {
        return values;
    }

    public BinaryDocValuesSub(MergeState.DocMap docMap, BinaryDocValues values) {
        super(docMap);
        if (values == null || (values.docID() != -1)) {
            throw new IllegalStateException("Doc values is either null or docID is not -1 ");
        }
        this.values = values;
    }

    @Override
    public int nextDoc() throws IOException {
        return values.nextDoc();
    }
}