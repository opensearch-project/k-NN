/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import lombok.AllArgsConstructor;
import org.opensearch.knn.plugin.stats.KNNFlag;

import java.util.function.Supplier;

/**
 * Supplier to determine whether field with particular k-NN engine has been built
 */
@AllArgsConstructor
public class FieldWithEngineSupplier implements Supplier<Boolean> {

    private final KNNFlag builtWithEngineFlag;

    @Override
    public Boolean get() {
        return builtWithEngineFlag.isValue();
    }
}
