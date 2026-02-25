/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.Setter;
import lombok.experimental.Accessors;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

@Accessors(fluent = true, chain = true)
@Setter
class NonNestedNMappingSchema {
    private String knnFieldName;
    private int dimension;
    private VectorDataType dataType;
    private String methodParamString;
    private String filterFieldName;
    private String idFieldName;
    private SpaceType spaceType;
    private Mode mode;
    private CompressionLevel compressionLevel;

    static String createModeCompressionLevelPartInMapping(final Mode mode, final CompressionLevel compressionLevel) {
        if (mode == Mode.NOT_CONFIGURED && compressionLevel == CompressionLevel.NOT_CONFIGURED) {
            return "";
        }

        return """
              "mode": "%s",
              "compression_level": "%s",
            """.formatted(mode == Mode.NOT_CONFIGURED ? "null" : mode.getName(), compressionLevel.getName());
    }

    public String createString() {
        final String mapping = """
            {
              "properties": {
                "%s": {
                  "type": "knn_vector",
                  "dimension": %s,
                  "data_type": "%s",
                  %s
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
            createModeCompressionLevelPartInMapping(mode, compressionLevel),
            spaceType.getValue(),
            methodParamString,
            filterFieldName,
            idFieldName
        ).stripIndent();

        return mapping;
    }
}
