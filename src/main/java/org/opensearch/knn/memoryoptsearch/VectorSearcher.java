/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
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
     * Returns the {@link ByteVectorValues} from the searcher. The behavior is undefined if
     * searcher doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     */
    ByteVectorValues getByteVectorValues() throws IOException;
}
