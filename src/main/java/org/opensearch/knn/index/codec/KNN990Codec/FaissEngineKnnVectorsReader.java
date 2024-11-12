/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;

import java.io.Closeable;
import java.io.IOException;

/** Reads vectors from faiss index. like Lucene KnnVectorsReader without search */
public abstract class FaissEngineKnnVectorsReader implements Closeable {

    /** Sole constructor */
    protected FaissEngineKnnVectorsReader() {}

    /**
     * Checks consistency of this reader.
     *
     * <p>Note that this may be costly in terms of I/O, e.g. may involve computing a checksum value
     * against large data files.
     */
    public abstract void checkIntegrity() throws IOException;

    /**
     * Returns the {@link FloatVectorValues} for the given {@code field}. The behavior is undefined if
     * the given field doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     */
    public abstract FloatVectorValues getFloatVectorValues(String field) throws IOException;

    /**
     * Returns the {@link ByteVectorValues} for the given {@code field}. The behavior is undefined if
     * the given field doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     */
    public abstract ByteVectorValues getByteVectorValues(String field) throws IOException;

    /**
     * Return true if and only if we can get native engine files and extract docValues.
     * @param field KNN vectors enabled on its {@link FieldInfo}
     * @return boolean for native engines vectors
     */
    public abstract boolean isNativeVectors(String field);
}
