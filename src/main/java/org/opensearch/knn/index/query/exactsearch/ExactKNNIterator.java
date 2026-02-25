/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import java.io.IOException;

interface ExactKNNIterator {
    int nextDoc() throws IOException;

    float score();
}
