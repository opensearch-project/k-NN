/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.binary;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

/**
 * Base class for binary index which inherits {@link FaissIndex}.
 * It has one additional meta info in header, `codeSize` which refers to the size of quantized vector.
 * For example, 8x compression applied to 100 dimension float vector would get 50 byte as a `codeSize`. (e.g. 50 = (4 * 100) / 8)
 */
@Getter
public abstract class FaissBinaryIndex extends FaissIndex {
    // Number of bytes per vector (e.g. dimension / 8)
    protected int codeSize;

    public FaissBinaryIndex(final String indexType) {
        super(indexType);
    }

    /**
     * Read common header for binary format.
     * FYI Faiss - <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1284">...</a>
     *
     * @param inputStream Input stream reading bytes from Faiss index file.
     * @throws IOException
     */
    protected void readBinaryCommonHeader(final IndexInput inputStream) throws IOException {
        dimension = inputStream.readInt();
        codeSize = inputStream.readInt();
        totalNumberOfVectors = Math.toIntExact(inputStream.readLong());

        // Consume `is_trained`, which is always true.
        inputStream.readByte();

        // Consume `metric type`. We don't rely on this metric type as internally,
        // as all distance calculation will be done with hamming distance calculator.
        inputStream.readInt();

        // Binary index always uses hamming space type.
        spaceType = SpaceType.HAMMING;
    }
}
