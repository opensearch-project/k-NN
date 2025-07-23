/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.QuantizedKNNBinaryVectorValues;

import java.util.function.Supplier;

public class QuantizedKNNVectorValuesSupplier implements Supplier<KNNVectorValues<?>> {
    private final Supplier<KNNVectorValues<?>> orgSupplier;
    private final BuildIndexParams indexInfo;

    public QuantizedKNNVectorValuesSupplier(final BuildIndexParams indexInfo) {
        this.orgSupplier = indexInfo.getKnnVectorValuesSupplier();
        this.indexInfo = indexInfo;
    }

    @Override
    public KNNVectorValues<?> get() {
        return new QuantizedKNNBinaryVectorValues(orgSupplier.get(), indexInfo);
    }
}
