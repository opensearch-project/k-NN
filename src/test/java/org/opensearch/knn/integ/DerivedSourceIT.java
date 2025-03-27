/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.knn.DerivedSourceUtils;
import java.util.List;

/**
 * Integration tests for derived source feature for vector fields. Currently, with derived source, there are
 * a few gaps in functionality. Ignoring tests for now as feature is experimental.
 */
public class DerivedSourceIT extends DerivedSourceTestCase {

    @SneakyThrows
    public void testFlatFields() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testObjectField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getObjectIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testNestedField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getNestedIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Single method for running end to end tests for different index configurations for derived source. In general,
     * flow of operations are
     *
     * @param indexConfigContexts {@link DerivedSourceUtils.IndexConfigContext}
     */
    @SneakyThrows
    private void testDerivedSourceE2E(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        assertEquals(6, indexConfigContexts.size());

        // Prepare the indices by creating them and ingesting data into them
        prepareOriginalIndices(indexConfigContexts);

        // Merging
        testMerging(indexConfigContexts);

        // Update. Skipping update tests for nested docs for now. Will add in the future.
        testUpdate(indexConfigContexts);

        // Delete
        testDelete(indexConfigContexts);

        // Search
        testSearch(indexConfigContexts);

        // Reindex
        testReindex(indexConfigContexts);
    }

}
