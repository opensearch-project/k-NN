/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.Getter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;

public class FlatVectorsReaderWithFieldName {

    @Getter
    private final FlatVectorsReader flatVectorsReader;
    @Getter
    private final String field;

    public FlatVectorsReaderWithFieldName(FlatVectorsReader reader, String field) {
        this.flatVectorsReader = reader;
        this.field = field;
    }
}
