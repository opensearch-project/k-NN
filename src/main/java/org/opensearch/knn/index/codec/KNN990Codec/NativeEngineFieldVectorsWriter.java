/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.Getter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * NativeEngineVectorFieldsWriter is a class that will be used to accumulate all the vectors during ingestion before
 * lucene does a flush. This class ensures that KNNVectorWriter is free from generics and this class can encapsulate
 * all the details related to vectors types and docIds.
 *
 * @param <T> float[] or byte[]
 */
class NativeEngineFieldVectorsWriter<T> extends KnnFieldVectorsWriter<T> {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeEngineFieldVectorsWriter.class);
    @Getter
    private final FieldInfo fieldInfo;
    /**
     * We are using a map here instead of list, because for sampler interface for quantization we have to advance the iterator
     * to a specific docId, there a list cannot be useful because a docId != index of the vector in the list. Similar
     * thing is true when we have vector field in child document. There doc Ids will not be consistent. Hence, we need to
     * use the map here.
     */
    @Getter
    private final Map<Integer, T> vectors;
    private int lastDocID = -1;
    private final InfoStream infoStream;
    @Getter
    private final FlatFieldVectorsWriter<T> flatFieldVectorsWriter;

    @SuppressWarnings("unchecked")
    static NativeEngineFieldVectorsWriter<?> create(
        final FieldInfo fieldInfo,
        final FlatFieldVectorsWriter<?> flatFieldVectorsWriter,
        final InfoStream infoStream
    ) {
        switch (fieldInfo.getVectorEncoding()) {
            case FLOAT32:
                return new NativeEngineFieldVectorsWriter<>(
                    fieldInfo,
                    (FlatFieldVectorsWriter<float[]>) flatFieldVectorsWriter,
                    infoStream
                );
            case BYTE:
                return new NativeEngineFieldVectorsWriter<>(fieldInfo, (FlatFieldVectorsWriter<byte[]>) flatFieldVectorsWriter, infoStream);
        }
        throw new IllegalStateException("Unsupported Vector encoding : " + fieldInfo.getVectorEncoding());
    }

    private NativeEngineFieldVectorsWriter(
        final FieldInfo fieldInfo,
        final FlatFieldVectorsWriter<T> flatFieldVectorsWriter,
        final InfoStream infoStream
    ) {
        this.fieldInfo = fieldInfo;
        this.infoStream = infoStream;
        vectors = new HashMap<>();
        this.flatFieldVectorsWriter = flatFieldVectorsWriter;
    }

    /**
     * Add new docID with its vector value to the given field for indexing. Doc IDs must be added in
     * increasing order.
     *
     * @param docID int
     * @param vectorValue T
     */
    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
        if (docID == lastDocID) {
            throw new IllegalArgumentException(
                "[NativeEngineKNNVectorWriter]VectorValuesField \""
                    + fieldInfo.name
                    + "\" appears more than once in this document (only one value is allowed per field)"
            );
        }
        // TODO: we can build the graph here too iteratively. but right now I am skipping that as we need iterative
        // graph build support on the JNI layer.
        assert docID > lastDocID;
        // ensuring that vector is provided to flatFieldWriter.
        flatFieldVectorsWriter.addValue(docID, vectorValue);
        vectors.put(docID, vectorValue);
        lastDocID = docID;
    }

    /**
     * Used to copy values being indexed to internal storage.
     *
     * @param vectorValue an array containing the vector value to add
     * @return a copy of the value; a new array
     */
    @Override
    public T copyValue(T vectorValue) {
        throw new UnsupportedOperationException("NativeEngineVectorFieldsWriter doesn't support copyValue operation");
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatFieldVectorsWriter.getDocsWithFieldSet().ramBytesUsed() + (long) this.vectors.size()
            * (long) (RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER) + (long) this.vectors.size()
                * RamUsageEstimator.shallowSizeOfInstance(Integer.class) + (long) vectors.size() * fieldInfo.getVectorDimension()
                    * fieldInfo.getVectorEncoding().byteSize + flatFieldVectorsWriter.ramBytesUsed();
    }
}
