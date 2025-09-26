/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.test.OpenSearchTestCase;

public class KNNQueryTests extends OpenSearchTestCase {

    public void getQueryDimensions() {
        KNNQuery query = KNNQuery.builder().queryVector(new float[] { 1.0f, 2.0f, 3.0f }).build();
        assertEquals(3, query.getQueryDimension());
        query = KNNQuery.builder().byteQueryVector(new byte[] { 0, 1 }).build();
        assertEquals(2, query.getQueryDimension());
    }
}
