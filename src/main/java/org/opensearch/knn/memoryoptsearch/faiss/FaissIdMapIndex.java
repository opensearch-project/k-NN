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
    private AbstractFaissHNSWIndex nestedIndex;
    private long[] vectorIdToDocIdMapping;

    public FaissIdMapIndex() {
        super(IXMP);
    }

    /**
     * Partially load id mapping and its nested index to which vector searching will be delegated.
     * Faiss deserialization code :
     * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1088">IndexIDMap.h</a>
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        readCommonHeader(input);
        final FaissIndex nestedIndex = FaissIndex.load(input);

        if (nestedIndex instanceof AbstractFaissHNSWIndex nestedHnswIndex) {
            this.nestedIndex = nestedHnswIndex;
        } else {
            throw new IllegalStateException(
                "Invalid nested index. Expected " + FaissHNSWIndex.class.getSimpleName() + " , but got " + nestedIndex.getIndexType()
            );
        }

        // Load `idMap`
        final long numElements = input.readLong();
        // This is a mapping table that maps internal vector id to Lucene document id.
        // In dense case where all documents having at least one KNN field which also has exactly one vector, we don't need this.
        // However, in a sparse case where not all documents having KNN field, we need this mapping table to convert inner vector id to
        // Lucene document id.
        // Another case is parent-child nested case. In which, this mapping table will map internal vector id to parent document id.
        final long[] vectorIdToDocIdMapping = new long[(int) numElements];
        input.readLongs(vectorIdToDocIdMapping, 0, vectorIdToDocIdMapping.length);

        // If `idMap` is an identity function that maps `i` to `i`, then we don't need to keep it.
        for (int i = 0; i < vectorIdToDocIdMapping.length; i++) {
            if (vectorIdToDocIdMapping[i] != i) {
                // Only keep it if it's not an identify mapping.
                this.vectorIdToDocIdMapping = vectorIdToDocIdMapping;
                break;
            }
        }
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return nestedIndex.getVectorEncoding();
    }

    @Override
    public String getIndexType() {
        return IXMP;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        if (vectorIdToDocIdMapping == null) {
            // Handle 'dense' case where all documents have at least one KNN field, which has exactly one vector.
            // No re-mapping is required.
            return nestedIndex.getFloatValues(indexInput);
        }

        // Re-mapping is required.
        return sparseFloatValues(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        if (vectorIdToDocIdMapping == null) {
            // Handle 'dense' case where all documents have at least one KNN field, which has exactly one vector.
            // No re-mapping is required.
            return nestedIndex.getByteValues(indexInput);
        }

        // Re-mapping is required.
        return sparseByteValues(indexInput);
    }

    /**
     * For sparse or nested cases, {@link ByteVectorValues} needs to be wrapped to correctly map an internal vector ID to a
     * Lucene document ID.
     *
     * @param indexInput An read stream to FAISS index file.
     * @return {@link ByteVectorValues} which is a byte vector random accessor.
     * @throws IOException
     */
    private ByteVectorValues sparseByteValues(IndexInput indexInput) throws IOException {
        final ByteVectorValues vectorValues = nestedIndex.getByteValues(indexInput);

        @RequiredArgsConstructor
        class SparseByteVectorValuesImpl extends ByteVectorValues {
            private final ByteVectorValues vectorValues;

            @Override
            public byte[] vectorValue(int internalVectorId) throws IOException {
                return vectorValues.vectorValue(internalVectorId);
            }

            @Override
            public int dimension() {
                return vectorValues.dimension();
            }

            public int ordToDoc(int internalVectorId) {
                // Convert an internal vector id to Lucene document id.
                return (int) vectorIdToDocIdMapping[internalVectorId];
            }

            public Bits getAcceptOrds(final Bits acceptDocs) {
                if (acceptDocs != null) {
                    final Bits internalBits = vectorValues.getAcceptOrds(acceptDocs);

                    return new Bits() {
                        @Override
                        public boolean get(int internalVectorId) {
                            // Apply filtering with a converted Lucene document id.
                            return internalBits.get((int) vectorIdToDocIdMapping[internalVectorId]);
                        }

                        @Override
                        public int length() {
                            return internalBits.length();
                        }
                    };
                }

                return null;
            }

            @Override
            public int size() {
                // The number of vectors
                return vectorValues.size();
            }

            @Override
            public ByteVectorValues copy() throws IOException {
                return new SparseByteVectorValuesImpl(vectorValues.copy());
            }
        }

        return new SparseByteVectorValuesImpl(vectorValues);
    }

    /**
     * For sparse or nested cases, {@link ByteVectorValues} needs to be wrapped to correctly map an internal vector ID to a
     * Lucene document ID.
     *
     * @param indexInput An read stream to FAISS index file.
     * @return {@link FloatVectorValues} which is a float vector random accessor.
     * @throws IOException
     */
    private FloatVectorValues sparseFloatValues(IndexInput indexInput) throws IOException {
        final FloatVectorValues vectorValues = nestedIndex.getFloatValues(indexInput);

        @RequiredArgsConstructor
        class SparseFloatVectorValuesImpl extends FloatVectorValues {
            private final FloatVectorValues vectorValues;

            @Override
            public float[] vectorValue(int internalVectorId) throws IOException {
                return vectorValues.vectorValue(internalVectorId);
            }

            @Override
            public int dimension() {
                return vectorValues.dimension();
            }

            public int ordToDoc(int internalVectorId) {
                // Convert an internal vector id to Lucene document id.
                return (int) vectorIdToDocIdMapping[internalVectorId];
            }

            public Bits getAcceptOrds(final Bits acceptDocs) {
                if (acceptDocs != null) {
                    final Bits internalBits = vectorValues.getAcceptOrds(acceptDocs);

                    return new Bits() {
                        @Override
                        public boolean get(int internalVectorId) {
                            // Apply filtering with a converted Lucene document id.
                            return internalBits.get((int) vectorIdToDocIdMapping[internalVectorId]);
                        }

                        @Override
                        public int length() {
                            return internalBits.length();
                        }
                    };
                }

                return null;
            }

            @Override
            public int size() {
                return vectorValues.size();
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                return new SparseFloatVectorValuesImpl(vectorValues.copy());
            }
        }

        return new SparseFloatVectorValuesImpl(vectorValues);
    }
}
