/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.FlatVectorsReaderWithFieldName;

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
    private FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName;

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
    protected void doLoad(IndexInput input, FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName) throws IOException {
        this.flatVectorsReaderWithFieldName = flatVectorsReaderWithFieldName;
        boolean dedupApplied = readCommonHeader(input);

        if (dedupApplied) {
            floatVectors = null;
        } else {
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
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    /**
     * Returns a {@link FloatVectorValues} view for reading vectors.
     * <p>
     * If deduplication is enabled, vectors will be read from the .vec file
     * instead of the flat vector section in the .faiss file.
     */
    @Override
    public FloatVectorValues getFloatValues(final IndexInput indexInput) throws IOException {
        if (floatVectors == null) {
            return flatVectorsReaderWithFieldName.getFlatVectorsReader().getFloatVectorValues(flatVectorsReaderWithFieldName.getField());
        }
        @RequiredArgsConstructor
        class FloatVectorValuesImpl extends FloatVectorValues {
            final IndexInput indexInput;
            final float[] buffer = new float[dimension];

            @Override
            public float[] vectorValue(int internalVectorId) throws IOException {
                indexInput.seek(floatVectors.getBaseOffset() + internalVectorId * oneVectorByteSize);
                indexInput.readFloats(buffer, 0, buffer.length);
                return buffer;
            }

            @Override
            public FloatVectorValues copy() {
                return new FloatVectorValuesImpl(indexInput.clone());
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

        return new FloatVectorValuesImpl(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support " + ByteVectorValues.class.getSimpleName());
    }
}
