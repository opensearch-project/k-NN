/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.stats.KNNFlag;

public class FieldWithEngineSupplierTests extends KNNTestCase {

    public void testSupplierCreateAndSet() {
        FieldWithEngineSupplier fieldWithEngineSupplier = new FieldWithEngineSupplier(KNNFlag.BUILT_WITH_FAISS);
        assertFalse(fieldWithEngineSupplier.get());
        KNNFlag.BUILT_WITH_FAISS.setValue(true);
        assertTrue(fieldWithEngineSupplier.get());
    }
}
