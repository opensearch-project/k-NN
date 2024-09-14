/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import junit.framework.TestCase;
import org.apache.lucene.index.SegmentInfo;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.List;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.calculateArraySize;

public class KNNCodecUtilTests extends TestCase {

    public void testCalculateArraySize() {
        int numVectors = 4;
        int vectorLength = 10;

        // Float data type
        VectorDataType vectorDataType = VectorDataType.FLOAT;
        assertEquals(160, calculateArraySize(numVectors, vectorLength, vectorDataType));

        // Byte data type
        vectorDataType = VectorDataType.BYTE;
        assertEquals(40, calculateArraySize(numVectors, vectorLength, vectorDataType));

        // Binary data type
        vectorDataType = VectorDataType.BINARY;
        assertEquals(40, calculateArraySize(numVectors, vectorLength, vectorDataType));
    }

    public void testGetKNNEngines() {
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        KNNEngine knnEngine = KNNEngine.FAISS;
        Set<String> SEGMENT_MULTI_FIELD_FILES_FAISS = Set.of("_0.cfe", "_0_2011_long_target_field.faissc", "_0_2011_target_field.faissc");
        when(segmentInfo.getUseCompoundFile()).thenReturn(true);
        when(segmentInfo.files()).thenReturn(SEGMENT_MULTI_FIELD_FILES_FAISS);
        List<String> engineFiles = KNNCodecUtil.getEngineFiles(knnEngine.getExtension(), "target_field", segmentInfo);
        assertEquals(engineFiles.size(), 2);
        assertTrue(engineFiles.get(0).equals("_0_2011_target_field.faissc"));
    }
}
