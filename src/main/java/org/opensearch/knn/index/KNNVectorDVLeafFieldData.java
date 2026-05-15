/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.fielddata.LeafFieldData;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.fielddata.SortedBinaryDocValues;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.search.DocValueFormat;

import java.io.IOException;

/**
 * Per-segment leaf field data for KNN vector fields. Provides access to vector values
 * for scripting ({@code doc['field']}) and {@code docvalue_fields} in search responses.
 *
 * <p><b>Lifecycle and threading:</b> A new instance is created per segment per search request
 * via {@link KNNVectorIndexFieldData#load(org.apache.lucene.index.LeafReaderContext)}.
 * Each search thread owns its own instance — there is no sharing across threads.
 *
 * <p><b>Iterator isolation:</b> Each call to {@link #getLeafValueFetcher(DocValueFormat)}
 * creates its own {@link KNNVectorValues} iterator, ensuring that multiple fetchers
 * (if ever created from the same instance) cannot corrupt each other's iteration state.
 */
public class KNNVectorDVLeafFieldData implements LeafFieldData {

    private final LeafReader reader;
    private final String fieldName;
    private final VectorDataType vectorDataType;

    public KNNVectorDVLeafFieldData(LeafReader reader, String fieldName, VectorDataType vectorDataType) {
        this.reader = reader;
        this.fieldName = fieldName;
        this.vectorDataType = vectorDataType;
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public long ramBytesUsed() {
        return 0; // unknown
    }

    @Override
    public ScriptDocValues<?> getScriptValues() {
        try {
            FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, fieldName);
            if (fieldInfo == null) {
                return KNNVectorScriptDocValues.emptyValues(fieldName, vectorDataType);
            }
            KnnVectorValues knnVectorValues;
            if (fieldInfo.hasVectorValues()) {
                switch (fieldInfo.getVectorEncoding()) {
                    case FLOAT32:
                        knnVectorValues = reader.getFloatVectorValues(fieldName);
                        break;
                    case BYTE:
                        knnVectorValues = reader.getByteVectorValues(fieldName);
                        break;
                    default:
                        throw new IllegalStateException("Unsupported Lucene vector encoding: " + fieldInfo.getVectorEncoding());
                }
                return KNNVectorScriptDocValues.create(knnVectorValues, fieldName, vectorDataType);
            }
            DocIdSetIterator values = DocValues.getBinary(reader, fieldName);
            return KNNVectorScriptDocValues.create(values, fieldName, vectorDataType);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load values for knn vector field: " + fieldName, e);
        }
    }

    @Override
    public SortedBinaryDocValues getBytesValues() {
        throw new UnsupportedOperationException("knn vector field '" + fieldName + "' doesn't support sorting");
    }

    /**
     * Returns a {@link DocValueFetcher.Leaf} that reads vector values directly from the
     * KNN vector index, bypassing {@code _source} entirely.
     *
     * <p>This powers {@code "docvalue_fields": ["my_vector_field"]} in search requests.
     * Unlike the k-NN DerivedSource path (which deserializes the entire {@code _source}, injects
     * the vector, and re-serializes), this reads the vector with a single seek — zero
     * {@code _source} parsing overhead.
     *
     * <p><b>Iterator isolation:</b> Each invocation creates a fresh {@link KNNVectorValues}
     * iterator so that multiple {@code Leaf} instances obtained from the same
     * {@code KNNVectorDVLeafFieldData} cannot interfere with each other's state.
     *
     * <p><b>Return type:</b> {@code float[]} for FLOAT vectors —
     * {@link org.opensearch.core.xcontent.XContentBuilder} serializes this as a JSON numeric array.
     *
     * @param format the doc value format — currently unused but required by the interface
     * @return a leaf fetcher that yields vector values per document, or an empty fetcher
     *         if the field has no vectors in this segment
     * @throws UnsupportedOperationException if the vector data type is BYTE or BINARY
     */
    @Override
    public DocValueFetcher.Leaf getLeafValueFetcher(final DocValueFormat format) {
        if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
            throw new UnsupportedOperationException(
                "docvalue_fields is not supported for [" + vectorDataType + "] vector field '" + fieldName + "'"
            );
        }

        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, fieldName);
        // This is important because if for a segment there is no vector field present and then customer still ask for
        // the vector field values for the docs in that segment, we should return empty array since vector are not present.
        if (fieldInfo == null) {
            return EMPTY_DOCVALUE_FETCHER_LEAF;
        }

        final KNNVectorValues<?> vectorValues;
        try {
            vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, Lucene.segmentReader(reader));
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load vector values for field: " + fieldName, e);
        }

        return new DocValueFetcher.Leaf() {
            private int count;

            @Override
            public boolean advanceExact(int docId) throws IOException {
                if (vectorValues.advance(docId) == docId) {
                    count = 1;
                    return true;
                }
                count = 0;
                return false;
            }

            @Override
            public int docValueCount() {
                return count;
            }

            @Override
            public Object nextValue() throws IOException {
                return vectorValues.conditionalCloneVector();
            }
        };
    }

    private static final DocValueFetcher.Leaf EMPTY_DOCVALUE_FETCHER_LEAF = new DocValueFetcher.Leaf() {

        @Override
        public boolean advanceExact(int docId) {
            return false;
        }

        @Override
        public int docValueCount() {
            return 0;
        }

        @Override
        public Object nextValue() {
            return null;
        }
    };
}
