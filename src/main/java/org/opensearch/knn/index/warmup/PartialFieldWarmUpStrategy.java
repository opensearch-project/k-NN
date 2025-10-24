/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Partially warm up the index by doing a search with an empty query vector
 */
public class PartialFieldWarmUpStrategy extends FieldWarmUpStrategy {
    private final SegmentReader segmentReader;

    public PartialFieldWarmUpStrategy(LeafReader leafReader) {
        this.segmentReader = Lucene.segmentReader(leafReader);
    }

    @Override
    public boolean warmUp(FieldInfo field) throws IOException {
        final String dataTypeStr = field.getAttribute(VECTOR_DATA_TYPE_FIELD);

        if (dataTypeStr == null) {
            return false;
        }

        final VectorDataType vectorDataType = VectorDataType.get(dataTypeStr);
        if (vectorDataType == VectorDataType.FLOAT) {
            segmentReader.getVectorReader().search(field.getName(), (float[]) null, null, null);
        } else {
            segmentReader.getVectorReader().search(field.getName(), (byte[]) null, null, null);
        }
        return true;
    }
}
