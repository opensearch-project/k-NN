/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.opensearch.knn.index.codec.nativeindex.IndexBuildSetup;
import org.opensearch.knn.index.codec.nativeindex.QuantizationIndexUtils;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

/**
 * This class is designed for uploading quantized binary vectors as part of the GPU indexing pipeline. Internally, we store the
 * full-precision vectors. When needed, it retrieves these full-precision vectors, applies quantization based on its current state, and
 * returns the quantized data as a byte[]. As a result, the `bytesPerVector` and `dimension` exposed externally refer to the quantized
 * (binary) vectors.
 * For instance, if the original vector has 768 dimensions and a 32× compression ratio is applied, the quantized vector size (code_size)
 * becomes 96 bytes. In that case, `bytesPerVector` would be set to `96`, and the `dimension` would be set to 768 bits. (e.g. one float
 * value became one bit)
 */
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
        return getVector();
    }
}
