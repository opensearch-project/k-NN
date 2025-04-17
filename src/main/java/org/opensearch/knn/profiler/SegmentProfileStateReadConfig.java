/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.index.SegmentReadState;

@Getter
@AllArgsConstructor
public class SegmentProfileStateReadConfig {
    private SegmentReadState segmentReadState;
    private String field;
}
