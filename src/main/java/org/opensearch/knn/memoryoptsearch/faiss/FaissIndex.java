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
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

/**
 * A FAISS index file consists of multiple logical sections, each beginning with four bytes indicating an index type. A section may contain
 * a nested section or vector storage, forming a tree structure with a top-level index as the starting point.
 * For example, for `float[]` vectors with the L2 metric, the top-level index is `IxMp`, which contains an HNSW graph as a nested index
 * (`IHNf`). The graph, in turn, uses flat vectors as storage (`IHNf`).
 * <p>
 * FYI : Faiss <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/Index.h">index.h</a>
 */
@Getter
public abstract class FaissIndex {
    // Index type name
    protected String indexType;
    // Vector dimension
    protected int dimension;
    // Total number of vectors saved within this index.
    protected int totalNumberOfVectors;
    // Space type used to index vectors in this index.
    protected SpaceType spaceType;

    public FaissIndex(final String indexType) {
        this.indexType = indexType;
    }

    /**
     * Recursively parse sections in a FAISS index file and map each section to a {@link FaissIndex}.
     * If an unsupported section is encountered, {@link UnsupportedFaissIndexException} will be thrown.
     * The first four bytes of each section represent an index type name. The index type is read first,
     * and {@link IndexTypeToFaissIndexMapping} is then used to delegate section loading to the corresponding {@link FaissIndex} subtype
     * implementation.
     *
     * @param input Input stream to a FAISS index
     * @return Top level {@link FaissIndex}.
     * @throws IOException
     */
    public static FaissIndex load(IndexInput input) throws IOException {
        final String indexType = readIndexType(input);
        final FaissIndex faissIndex = IndexTypeToFaissIndexMapping.getFaissIndex(indexType);
        faissIndex.doLoad(input);
        return faissIndex;
    }

    protected abstract void doLoad(IndexInput input) throws IOException;

    protected void readCommonHeader(IndexInput readStream) throws IOException {
        dimension = readStream.readInt();
        totalNumberOfVectors = Math.toIntExact(readStream.readLong());
        // consume 2 dummy deprecated fields.
        readStream.readLong();
        readStream.readLong();

        // We don't use this field
        final boolean isTrained = readStream.readByte() == 1;

        final int metricTypeIndex = readStream.readInt();
        if (metricTypeIndex == 0) {
            spaceType = SpaceType.INNER_PRODUCT;
        } else if (metricTypeIndex == 1) {
            spaceType = SpaceType.L2;
        } else {
            throw new IllegalStateException("Partial loading does not support metric type index=" + metricTypeIndex + " from FAISS.");
        }
    }

    static private String readIndexType(final IndexInput input) throws IOException {
        final byte[] fourBytes = new byte[4];
        input.readBytes(fourBytes, 0, fourBytes.length);
        return new String(fourBytes);
    }

    /**
     * Returns the vector encoding scheme indicating how vectors are stored internally, either as Byte or Float.
     *
     * @return Vector encoding, Byte or Float.
     */
    public abstract VectorEncoding getVectorEncoding();

    /**
     * A similarity function is required to calculate the similarity with a query vector.
     *
     * @return Similarity function.
     */
    public KNNVectorSimilarityFunction getVectorSimilarityFunction() {
        return spaceType.getKnnVectorSimilarityFunction();
    }

    /**
     * This method returns an accessor for random float vectors.
     *
     * @param indexInput Input stream to a FAISS index file.
     * @return Float vector random accessor.
     * @throws IOException
     */
    public abstract FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException;

    /**
     * This method returns an accessor for random byte vectors.
     *
     * @param indexInput Input stream to a FAISS index file
     * @return Byte vector random accessor.
     * @throws IOException
     */
    public abstract ByteVectorValues getByteValues(IndexInput indexInput) throws IOException;
}
