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
public class NestedMappingSchema {
    private String nestedFieldName;
    private String knnFieldName;
    private int dimension;
    private VectorDataType dataType;
    private String methodParamString;
    private String filterFieldName;
    private String idFieldName;
    private SpaceType spaceType;

    public String createString() {
        final String mapping = """
            {
              "properties": {
                "%s": {
                  "type": "nested",
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
                    }
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
            }
            """.formatted(
            nestedFieldName,
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
