/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * A FAISS index with an ID mapping that maps the internal vector ID to a logical ID, along with the actual vector index.
 * It first delegates the vector search to its nested vector index, then transforms the vector ID into a logical index that is
 * understandable by upstream components. This is particularly useful when not all Lucene documents are indexed with a vector field.
 * For example, if 70% of the documents have a vector field and the remaining 30% do not, the FAISS vector index will still assign
 * increasing and continuous vector IDs starting from 0.
 * However, these IDs only cover the sparse 30% of Lucene documents, so an ID mapping is needed to convert the internal physical vector ID
 * into the corresponding Lucene document ID.
 * If the mapping is an identity mapping, where each `i` is mapped to itself, we omit storing it to save memory.
 */
public class FaissIdMapIndex extends FaissIndex {
    public static final String IXMP = "IxMp";

    @Getter
    private FaissHNSWFlatIndex nestedIndex;
    private long[] vectorIdToDocIdMapping;
    private long oneVectorByteSize;

    /**
     * Partially load id mapping and its nested index to which vector searching will be delegated.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @return {@link FaissIdMapIndex} instance consists of index hierarchy.
     * @throws IOException
     */
    public static FaissIdMapIndex load(IndexInput input) throws IOException {
        FaissIdMapIndex faissIdMapIndex = new FaissIdMapIndex();
        readCommonHeader(input, faissIdMapIndex);
        FaissIndex nestedIndex = FaissIndex.load(input);
        if (nestedIndex instanceof FaissHNSWFlatIndex) {
            faissIdMapIndex.nestedIndex = (FaissHNSWFlatIndex) nestedIndex;
        } else {
            throw new IllegalStateException("Invalid nested index. Expected FaissHNSWFlatIndex, but got " + nestedIndex.getIndexType());
        }
        faissIdMapIndex.oneVectorByteSize = faissIdMapIndex.nestedIndex.getStorage().getOneVectorByteSize();

        // Load `idMap`
        final long numElements = input.readLong();
        long[] vectorIdToDocIdMapping = new long[(int) numElements];
        input.readLongs(vectorIdToDocIdMapping, 0, vectorIdToDocIdMapping.length);

        // If `idMap` is an identity function that maps `i` to `i`, then we don't need to keep it.
        for (int i = 0; i < vectorIdToDocIdMapping.length; i++) {
            if (vectorIdToDocIdMapping[i] != i) {
                // Only keep it if it's not an identify mapping.
                faissIdMapIndex.vectorIdToDocIdMapping = vectorIdToDocIdMapping;
                break;
            }
        }

        return faissIdMapIndex;
    }

    @Override
    public String getIndexType() {
        return IXMP;
    }

    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        if (vectorIdToDocIdMapping == null) {
            return denseFloatValues(indexInput);
        }

        return sparseFloatValues(indexInput);
    }

    private FloatVectorValues denseFloatValues(IndexInput indexInput) throws IOException {
        final FaissIndexFlat indexFlat = nestedIndex.getStorage();
        final Storage codes = indexFlat.getCodes();
        final int dimension = getDimension();
        final int totalNumVectors = (int) getTotalNumberOfVectors();

        @RequiredArgsConstructor
        class DenseFloatVectorValuesImpl extends FloatVectorValues {
            final IndexInput data;
            final float[] vector = new float[dimension];

            @Override
            public float[] vectorValue(int targetOrd) throws IOException {
                data.seek(oneVectorByteSize * targetOrd);
                data.readFloats(vector, 0, vector.length);
                return vector;
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public int size() {
                return totalNumVectors;
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                return new DenseFloatVectorValuesImpl(indexInput.slice("FaissIndexFlat", codes.baseOffset, codes.sectionSize));
            }
        }

        return new DenseFloatVectorValuesImpl(indexInput.slice("FaissIndexFlat", codes.baseOffset, codes.sectionSize));
    }

    private FloatVectorValues sparseFloatValues(IndexInput indexInput) throws IOException {
        final FaissIndexFlat indexFlat = nestedIndex.getStorage();
        final Storage codes = indexFlat.getCodes();
        final int dimension = getDimension();
        final int totalNumVectors = (int) getTotalNumberOfVectors();

        @RequiredArgsConstructor
        class SparseFloatVectorValuesImpl extends FloatVectorValues {
            final IndexInput data;
            final float[] vector = new float[dimension];

            @Override
            public float[] vectorValue(int targetOrd) throws IOException {
                data.seek(oneVectorByteSize * targetOrd);
                data.readFloats(vector, 0, vector.length);
                return vector;
            }

            @Override
            public int dimension() {
                return dimension;
            }

            public int ordToDoc(int ord) {
                return (int) vectorIdToDocIdMapping[ord];
            }

            public Bits getAcceptOrds(final Bits acceptDocs) {
                return acceptDocs == null ? null : new Bits() {
                    @Override
                    public boolean get(int ord) {
                        return acceptDocs.get((int) vectorIdToDocIdMapping[ord]);
                    }

                    @Override
                    public int length() {
                        return totalNumVectors;
                    }
                };
            }

            @Override
            public int size() {
                return totalNumVectors;
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                return new SparseFloatVectorValuesImpl(indexInput.slice("FaissIndexFlat", codes.baseOffset, codes.sectionSize));
            }
        }

        return new SparseFloatVectorValuesImpl(indexInput.slice("FaissIndexFlat", codes.baseOffset, codes.sectionSize));
    }
}
