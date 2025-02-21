/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import org.apache.lucene.index.VectorSimilarityFunction;

public class FaissIndexFlatIP extends FaissIndexFlat {

    @Override
    public VectorSimilarityFunction getSimilarityFunction() {
        return VectorSimilarityFunction.DOT_PRODUCT;
    }
}
