/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.opensearch.knn.generate.IndexingType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import static org.opensearch.knn.generate.DocumentsGenerator.DIMENSIONS;
import static org.opensearch.knn.generate.DocumentsGenerator.FILTER_FIELD_NAME;
import static org.opensearch.knn.generate.DocumentsGenerator.ID_FIELD_NAME;
import static org.opensearch.knn.generate.DocumentsGenerator.KNN_FIELD_NAME;

public class DiskMode1xCompressionSupportedIT extends AbstractMemoryOptimizedKnnSearchIT {
    @SneakyThrows
    public void testNoOffHeapIndexIsLoaded() {
        // Make mapping schema string.
        final NonNestedNMappingSchema mapping = new NonNestedNMappingSchema().knnFieldName(KNN_FIELD_NAME)
            .dimension(DIMENSIONS)
            .dataType(VectorDataType.FLOAT)
            .mode(Mode.ON_DISK)
            .compressionLevel(CompressionLevel.x1)
            .spaceType(SpaceType.INNER_PRODUCT)
            .filterFieldName(FILTER_FIELD_NAME)
            .idFieldName(ID_FIELD_NAME);

        final String mappingStr = mapping.createString();
        final Schema schema = new Schema(mappingStr, VectorDataType.FLOAT, Mode.ON_DISK, NO_ADDITIONAL_SETTINGS);

        // Start validate dense, sparse cases.
        doKnnSearchTest(SpaceType.INNER_PRODUCT, schema, IndexingType.DENSE, false, false, true);
        doKnnSearchTest(SpaceType.INNER_PRODUCT, schema, IndexingType.DENSE, false, false, false);

        // Even memory_opt_srch = false, it should have used LuceneOnFaiss regardless, no off-heap is expected.
        assertEquals(0, getTotalGraphsInCache());
    }
}
