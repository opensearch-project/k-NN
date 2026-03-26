/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;

import java.io.Closeable;
import java.io.IOException;

/**
 * This searcher performs vector search on non-Lucene index, for example FAISS index.
 * Two search APIs will be compatible with Lucene, taking {@link KnnCollector} and {@link Bits}.
 * In its implementation, it must collect top vectors that is similar to the given query. Make sure to transform the result to similarity
 * value if internally calculates distance between.
 *
 * <p>TODO: Refactor to eliminate this interface in favor of extending KnnVectorReader directly.
 * VectorSearcher duplicates functionality already present in KnnVectorReader (search methods and getByteVectorValues),
 * creating maintenance overhead and potential inconsistencies. Remove this interface and have all implementing classes
 * extend KnnVectorReader instead.
 */
public interface VectorSearcher extends Closeable {
    /**
     * Return the k nearest neighbor documents as determined by comparison of their vector values for
     * this field, to the given vector, by the field's similarity function. The score of each document
     * is derived from the vector similarity in a way that ensures scores are positive and that a
     * larger score corresponds to a higher ranking.
     *
     * <p>The search is allowed to be approximate, meaning the results are not guaranteed to be the
     * true k closest neighbors. For large values of k (for example when k is close to the total
     * number of documents), the search may also retrieve fewer than k documents.
     *
     * @param target the vector-valued float vector query
     * @param knnCollector a KnnResults collector and relevant settings for gathering vector results
     * @param acceptDocs {@link Bits} that represents the allowed documents to match, or {@code null}
     *     if they are all allowed to match.
     */
    void search(float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException;

    /**
     * Return the k nearest neighbor documents as determined by comparison of their vector values for
     * this field, to the given vector, by the field's similarity function. The score of each document
     * is derived from the vector similarity in a way that ensures scores are positive and that a
     * larger score corresponds to a higher ranking.
     *
     * <p>The search is allowed to be approximate, meaning the results are not guaranteed to be the
     * true k closest neighbors. For large values of k (for example when k is close to the total
     * number of documents), the search may also retrieve fewer than k documents.
     *
     * @param target the vector-valued byte vector query
     * @param knnCollector a KnnResults collector and relevant settings for gathering vector results
     * @param acceptDocs {@link Bits} that represents the allowed documents to match, or {@code null}
     *     if they are all allowed to match.
     */
    void search(byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException;

    /**
     * Returns the {@link ByteVectorValues} from the searcher, using the provided iterator
     * for the underlying scorer. When the returned values produce a {@link org.apache.lucene.search.VectorScorer},
     * that scorer will iterate using {@code iterator} instead of the default one.
     *
     * <p>Why a {@link DocIndexIterator} is required</p>
     * The FAISS-backed {@link ByteVectorValues} implementation is a random-access store: it can
     * look up any vector by ordinal, but it does not inherently know how to iterate over
     * documents. A {@link org.apache.lucene.search.VectorScorer}, however, needs a
     * {@link org.apache.lucene.search.DocIdSetIterator} to drive scoring. Because the FAISS
     * layer cannot construct one on its own, the iterator must be supplied externally.
     *
     * <p>Three approaches were considered:
     * <ol>
     *   <li><b>Pass the iterator when requesting {@code ByteVectorValues} (chosen approach)</b> –
     *       the caller provides a {@link DocIndexIterator} here, and the
     *       implementation wraps it into the scorer. This keeps the FAISS loading path
     *       unchanged and gives the caller full control over which iterator is used.</li>
     *   <li>Accept the iterator during the FAISS index load ({@code FaissIndex.load}) so it
     *       propagates to every {@code ByteVectorValues} instance automatically.</li>
     *   <li>Construct the iterator inside the {@code ByteVectorValues} implementation
     *       itself.</li>
     * </ol>
     *
     * <p>TODO: Clean this up by moving toward building the iterator/scorer inside the
     * {@code ByteVectorValues} implementation so callers no longer need to supply one.
     *
     * @param iterator the iterator the scorer should use for document traversal
     */
    ByteVectorValues getByteVectorValues(DocIndexIterator iterator) throws IOException;

    void warmUp() throws IOException;
}
