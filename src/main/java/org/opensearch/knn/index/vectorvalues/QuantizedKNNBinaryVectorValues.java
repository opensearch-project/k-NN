/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.opensearch.knn.index.codec.nativeindex.IndexBuildSetup;
import org.opensearch.knn.index.codec.nativeindex.QuantizationIndexUtils;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

public class QuantizedKNNBinaryVectorValues extends KNNVectorValues<byte[]> {
    private KNNFloatVectorValues knnFloatVectorValues;
    private IndexBuildSetup indexBuildSetup;

    public QuantizedKNNBinaryVectorValues(final KNNVectorValues<?> orgKnnVectorValues, final BuildIndexParams indexInfo) {
        super(extractIteratorSafeAndSet(orgKnnVectorValues));
        this.knnFloatVectorValues = (KNNFloatVectorValues) orgKnnVectorValues;
        this.indexBuildSetup = QuantizationIndexUtils.prepareIndexBuild(orgKnnVectorValues, indexInfo);
    }

    private static KNNVectorValuesIterator extractIteratorSafeAndSet(final KNNVectorValues<?> orgKnnVectorValues) {
        if ((orgKnnVectorValues instanceof KNNFloatVectorValues) == false) {
            throw new IllegalArgumentException(
                "Expected " + KNNFloatVectorValues.class.getName() + " but got " + orgKnnVectorValues.getClass().getSimpleName()
            );
        }

        return orgKnnVectorValues.vectorValuesIterator;
    }

    @Override
    public byte[] getVector() throws IOException {
        final byte[] quantizedVector = (byte[]) QuantizationIndexUtils.processAndReturnVector(knnFloatVectorValues, indexBuildSetup);
        this.dimension = quantizedVector.length * Byte.SIZE;
        this.bytesPerVector = quantizedVector.length;
        return quantizedVector;
    }

    @Override
    public byte[] conditionalCloneVector() throws IOException {
        throw new UnsupportedOperationException(
            "Cloning vector is not supported in " + QuantizedKNNBinaryVectorValues.class.getSimpleName()
        );
    }
}
