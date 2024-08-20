/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.model;

import lombok.Builder;
import lombok.ToString;
import lombok.Value;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Map;

@Value
@Builder
@ToString
public class BuildIndexParams {
    String fieldName;
    KNNEngine knnEngine;
    String indexPath;
    VectorDataType vectorDataType;
    Map<String, Object> parameters;
    // TODO: Add quantization state as parameter to build index
}
