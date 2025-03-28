/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.Setter;
import lombok.experimental.Accessors;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

@Accessors(fluent = true, chain = true)
@Setter
class NonNestedNMappingSchema {
    String knnFieldName;
    int dimension;
    VectorDataType dataType;
    String methodParamString;
    String filterFieldName;
    String idFieldName;
    SpaceType spaceType;

    public String createString() {
        final String mapping = """
            {
              "properties": {
                "%s": {
                  "type": "knn_vector",
                  "dimension": %s,
                  "data_type": "%s",
                  "space_type": "%s",
                  "method": {
                    "engine": "faiss",
                    "name": "hnsw",
                    "parameters": %s
                  }
                },
                "%s": {
                  "type": "keyword",
                  "index": true
                },
                "%s": {
                  "type": "keyword",
                  "index": true
                }
              },
              "dynamic": false
            }""".formatted(
            knnFieldName,
            Integer.toString(dimension),
            dataType.getValue(),
            spaceType.getValue(),
            methodParamString,
            filterFieldName,
            idFieldName
        ).stripIndent();

        return mapping;
    }
}
