/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
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
    // Section name written by Faiss when IO_FLAG_SKIP_STORAGE is set (e.g., BBQ skips flat vector storage).
    public static final String NULL_INDEX_TYPE = "null";

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
     * When a "null" section name is encountered (e.g., Faiss BBQ where storage was skipped via IO_FLAG_SKIP_STORAGE),
     * this method returns {@code null}. The caller is responsible for handling the null case
     * (e.g., by wiring in a {@code FaissBBQFlatIndex} backed by Lucene's quantized reader).
     *
     * @param input Input stream to a FAISS index
     * @return Top level {@link FaissIndex}, or {@code null} if a "null" section is encountered.
     * @throws IOException
     */
    public static FaissIndex load(IndexInput input) throws IOException {
        final String indexType = FaissIndexLoadUtils.readIndexType(input);
        if (NULL_INDEX_TYPE.equals(indexType)) {
            return null;
        }
        final FaissIndex faissIndex = IndexTypeToFaissIndexMapping.getFaissIndex(indexType);
        faissIndex.doLoad(input);
        return faissIndex;
    }

    /**
     * Loads a FAISS index and, if required based on the field's configuration, wires in the appropriate
     * flat vector storage via {@link FaissFlatIndexFactory}.
     *
     * @param input             Input stream to a FAISS index
     * @param fieldInfo         Field metadata used to determine the flat index type
     * @param flatVectorsReader Reader providing both the scorer and, for certain index types (e.g. BBQ),
     *                          the backing flat vector storage
     * @return Top level {@link FaissIndex}
     * @throws IOException
     */
    public static FaissIndex load(IndexInput input, FieldInfo fieldInfo, FlatVectorsReader flatVectorsReader) throws IOException {
        final FaissIndex faissIndex = load(input);
        maybeSetFlatIndex(faissIndex, fieldInfo, flatVectorsReader);
        return faissIndex;
    }

    // If the HNSW index has no flat storage (e.g. BBQ skips it via IO_FLAG_SKIP_STORAGE), wire in the appropriate flat index.
    private static void maybeSetFlatIndex(
        final FaissIndex faissIndex,
        final FieldInfo fieldInfo,
        final FlatVectorsReader flatVectorsReader
    ) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            final FaissIndex nested = idMapIndex.getNestedIndex();
            if (nested instanceof AbstractFaissHNSWIndex hnswIndex && hnswIndex.getFlatVectors() == null) {
                final FaissIndex flatIndex = FaissFlatIndexFactory.create(fieldInfo, flatVectorsReader);
                if (flatIndex != null) {
                    hnswIndex.flatVectors = flatIndex;
                }
            }
        }
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
