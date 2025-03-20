/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import static org.opensearch.knn.common.KNNConstants.*;

public class SegmentSizeFloorMergePolicyIT extends KNNRestTestCase {
    @SneakyThrows
    public void testKNNIndexFloorSegmentSize() {
        // Create a KNN index
        String knnIndexName = "test-knn-floor-segment";
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 3)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(knnIndexName, mapping);

        // Create a non-KNN index
        String regularIndexName = "test-regular-floor-segment";
        createIndex(regularIndexName, Settings.EMPTY);

        String knnFloorSegment = getIndexSettingByName(knnIndexName, "index.merge.policy.floor_segment", true);
        String regularFloorSegment = getIndexSettingByName(regularIndexName, "index.merge.policy.floor_segment", true);

        // Assert KNN index has 16mb floor segment
        assertEquals("KNN index should have 16mb floor segment", "16mb", knnFloorSegment);

        // Assert regular index has default 2mb floor segment
        assertEquals("Regular index should have 2mb floor segment", "2097152b", regularFloorSegment);

        // Clean up
        deleteKNNIndex(knnIndexName);
        deleteKNNIndex(regularIndexName);
    }
}
