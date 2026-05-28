/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.SpaceType;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Integration tests for radial search on flat quantized indices (method: flat, 1-bit SQ by default).
 * The flat method is engine-agnostic — it does not specify an engine explicitly.
 */
public class FlatSQRadialSearchIT extends AbstractRadialSearchOnQuantizedIndexIT {

    private static final String INDEX_NAME = "flat_sq_radial_search_test";

    @Override
    protected String getIndexName() {
        return INDEX_NAME;
    }

    @Override
    protected void createQuantizedIndex() throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("space_type", SpaceType.L2.getValue())
            .startObject("method")
            .field("name", "flat")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put(KNN_INDEX, true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    @Override
    protected void createQuantizedIndexWithMaxResultWindow(int maxResultWindow) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("space_type", SpaceType.L2.getValue())
            .startObject("method")
            .field("name", "flat")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put("index.max_result_window", maxResultWindow)
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }
}
