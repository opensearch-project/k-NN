/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;
import java.util.Map;
import java.util.function.Supplier;

/**
 * This index type represents the storage of flat float vectors in FAISS.
 * Each vector occupies a fixed size proportional to its dimension.
 * The total storage size is calculated as `4 * dimension * number_of_vectors`, where `4` is the size of a float.
 * Please refer to IndexFlatL2 and IndexFlatIp in <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexFlat.h">...</a>.
 */
public class FaissIndexFloatFlat extends FaissIndex {
    // Flat format for L2 metric
    public static final String IXF2 = "IxF2";
    // Flat format for inner product metric
    public static final String IXFI = "IxFI";

    private static final Map<String, Supplier<KNNVectorSimilarityFunction>> INDEX_TYPE_TO_INDEX_FLOAT_FLAT = Map.of(
        IXF2,
        () -> KNNVectorSimilarityFunction.EUCLIDEAN,
        IXFI,
        () -> KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
    );

    private FaissSection floatVectors;
    private long oneVectorByteSize;
    @Getter
    private final KNNVectorSimilarityFunction vectorSimilarityFunction;

    public FaissIndexFloatFlat(final String indexType) {
        super(indexType);

        vectorSimilarityFunction = INDEX_TYPE_TO_INDEX_FLOAT_FLAT.getOrDefault(indexType, () -> {
            throw new IllegalStateException("Faiss index float flat does not support the index type [" + indexType + "].");
        }).get();
    }

    /**
     * Partial load the flat float vector section which is dimension * sizeof(float) * total_number_of_vectors.
     * FYI FAISS - <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L537">...</a>
     *
     * @param input
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        readCommonHeader(input);
        oneVectorByteSize = (long) Float.BYTES * getDimension();
        floatVectors = new FaissSection(input, Float.BYTES);
        if (floatVectors.getSectionSize() != (getTotalNumberOfVectors() * oneVectorByteSize)) {
            throw new IllegalStateException(
                "Got an inconsistent bytes size of vector ["
                    + floatVectors.getSectionSize()
                    + "] "
                    + "when faissIndexFloatFlat.totalNumberOfVectors="
                    + getTotalNumberOfVectors()
                    + ", faissIndexFloatFlat.oneVectorByteSize="
                    + oneVectorByteSize
            );
        }
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(final IndexInput indexInput) {
        class FloatVectorValuesImpl extends FloatVectorValues {
            final float[] buffer = new float[dimension];

            @Override
            public float[] vectorValue(int i) throws IOException {
                indexInput.seek(floatVectors.getBaseOffset() + i * oneVectorByteSize);
                indexInput.readFloats(buffer, 0, buffer.length);
                return buffer;
            }

            @Override
            public FloatVectorValues copy() {
                return new FloatVectorValuesImpl();
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public int size() {
                return totalNumberOfVectors;
            }
        }

        return new FloatVectorValuesImpl();
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support " + ByteVectorValues.class.getSimpleName());
    }
}
