/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizedValueReconstructor;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizedValueReconstructorFactory;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Map;

/**
 * This index type represents for a flat vector storage that is quantized in multiple formats.
 * Similar to {@link FaissIndexFloatFlat}, each vector occupies a fixed size depending on the configured quantization type.
 * For example, the quantization type `QT_8BIT` indicates that each element in a vector is quantized into 8bits. Therefore, each element
 * will occupy exactly one byte, a vector would occupy exactly the size of dimensions.
 */
@Log4j2
@Getter
public class FaissIndexScalarQuantizedFlat extends FaissIndex {
    private static EnumMap<FaissQuantizerType, VectorEncoding> VECTOR_DATA_TYPES = new EnumMap<>(
        Map.of(FaissQuantizerType.QT_8BIT_DIRECT_SIGNED, VectorEncoding.BYTE, FaissQuantizerType.QT_FP16, VectorEncoding.FLOAT32)
    );

    public static final String IXSQ = "IxSQ";

    private FaissQuantizerType quantizerType;
    private FaissQuantizedValueReconstructor reconstructor;
    private RangeStat rangeStat;
    private float rangeStatArgument;
    private int dimension;
    private long oneVectorByteSize;
    private int oneVectorElementBits;
    private FaissSection trainedValues;
    private FaissSection flatVectors;
    private VectorEncoding vectorEncoding;

    public FaissIndexScalarQuantizedFlat() {
        super(IXSQ);
    }

    /**
     * Deserialize the section and load important quantization related information including common header.
     * Refer to <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L789">here</a>.
     *
     * @param input Input stream for FAISS index file.
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        readCommonHeader(input);

        // Load quantizer type
        quantizerType = FaissQuantizerType.values()[input.readInt()];
        if (VECTOR_DATA_TYPES.containsKey(quantizerType) == false) {
            throw new UnsupportedFaissIndexException("Unsupported quantizer type: " + quantizerType);
        }
        vectorEncoding = VECTOR_DATA_TYPES.get(quantizerType);

        // Loading range statistics + arguments
        // Although it won't be used for searching, as it's for training, keep them for debugging purposes.
        rangeStat = RangeStat.values()[input.readInt()];
        float[] singleFloat = new float[1];
        input.readFloats(singleFloat, 0, 1);
        rangeStatArgument = singleFloat[0];

        dimension = Math.toIntExact(input.readLong());

        // Read code size
        input.readLong();

        trainedValues = new FaissSection(input, Float.BYTES);
        setDerivedSizes();

        flatVectors = new FaissSection(input, Byte.BYTES);

        // This should be put at the last as it needs dimension info + etc.
        reconstructor = FaissQuantizedValueReconstructorFactory.create(quantizerType, dimension, oneVectorElementBits);
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return vectorEncoding;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) {
        @RequiredArgsConstructor
        final class FloatVectorValuesImpl extends FloatVectorValues {
            final IndexInput indexInput;
            final byte[] bytesBuffer = new byte[(int) oneVectorByteSize];
            final float[] floatBuffer = new float[dimension];

            @Override
            public float[] vectorValue(int internalVectorId) throws IOException {
                indexInput.seek(flatVectors.getBaseOffset() + internalVectorId * oneVectorByteSize);
                indexInput.readBytes(bytesBuffer, 0, bytesBuffer.length);
                reconstructor.reconstruct(bytesBuffer, floatBuffer);
                return floatBuffer;
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public int size() {
                return totalNumberOfVectors;
            }

            @Override
            public FloatVectorValuesImpl copy() {
                return new FloatVectorValuesImpl(indexInput.clone());
            }
        }

        return new FloatVectorValuesImpl(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) {
        @RequiredArgsConstructor
        final class ByteVectorValuesImpl extends ByteVectorValues {
            final IndexInput indexInput;
            final byte[] buffer = new byte[(int) oneVectorByteSize];

            @Override
            public byte[] vectorValue(int internalVectorId) throws IOException {
                indexInput.seek(flatVectors.getBaseOffset() + internalVectorId * oneVectorByteSize);
                indexInput.readBytes(buffer, 0, buffer.length);
                reconstructor.reconstruct(buffer, buffer);
                return buffer;
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public int size() {
                return totalNumberOfVectors;
            }

            @Override
            public ByteVectorValues copy() {
                return new ByteVectorValuesImpl(indexInput.clone());
            }
        }

        return new ByteVectorValuesImpl(indexInput);
    }

    @Override
    public String getIndexType() {
        return IXSQ;
    }

    private void setDerivedSizes() {
        switch (quantizerType) {
            case QT_8BIT:
            case QT_8BIT_UNIFORM:
            case QT_8BIT_DIRECT:
            case QT_8BIT_DIRECT_SIGNED:
                // Ex: For 100 dimensions, one vector would occupy 100 bytes.
                oneVectorByteSize = dimension;
                oneVectorElementBits = 8;
                break;

            case QT_4BIT:
            case QT_4BIT_UNIFORM:
                // Ex: For 100 dimensions, one vector would occupy 50 bytes.
                oneVectorByteSize = (dimension + 1) / 2;  // Equivalent to Ceil(dimension / 2)
                oneVectorElementBits = 4;
                break;

            case QT_6BIT:
                // Ex: For 100 dimensions, one vector would occupy 75 bytes.
                oneVectorByteSize = (dimension * 6 + 7) / 8;  // Equivalent to Ceil((dimension * 6) / 8)
                oneVectorElementBits = 6;
                break;

            case QT_FP16:
            case QT_BF16:
                // Ex: For 100 dimensions, one vector would occupy 200 bytes as one element is fp16 (2 bytes)
                oneVectorByteSize = dimension * 2;
                oneVectorElementBits = 16;
                break;
        }
    }

    // Refer to https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ScalarQuantizer.h#L47
    public enum RangeStat {
        // [min - rangeStatArgument * (max - min), max + rangeStatArgument * (max - min)]
        MIN_MAX,
        // [mean - std * rangeStatArgument, mean + std * rangeStatArgument]
        MEAN_STD,
        // [Q(rangeStatArgument), Q(1 - rangeStatArgument)]
        QUANTILES,
        // Alternate optimization of reconstruction error
        OPTIM
    }
}
