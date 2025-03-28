/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.generate;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Nested document JSON strings generator.
 * It will have the nested field to have three child documents.
 */
public class NestedDocumentsGenerator extends DocumentsGenerator {
    public NestedDocumentsGenerator(final int numDocuments, final IndexingType indexingType, final VectorDataType dataType) {
        super(numDocuments, indexingType, dataType);
    }

    @Override
    protected GeneratedDocAndVector generateOneDoc(int idNo) {
        try {
            final XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
            final List<float[]> vectors = new ArrayList<>();

            // Id field
            final String id = "id-" + idNo;
            builder.field(ID_FIELD_NAME, id);

            // Filtering field
            final String filter = "filter-" + (idNo % FILTER_ID_NO_MOD);
            builder.field(FILTER_FIELD_NAME, filter);

            // Nested field
            if (indexingType == IndexingType.DENSE_NESTED || ThreadLocalRandom.current().nextFloat() < DENSE_RATIO) {
                builder.startArray(NESTED_FIELD_NAME);
                for (int i = 0; i < NUM_CHILD_DOCS; ++i) {
                    generateOneChildDoc(builder, vectors);
                }
                builder.endArray();
            }

            final String docString = builder.endObject().toString();
            return new GeneratedDocAndVector(docString, vectors);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void generateOneChildDoc(final XContentBuilder builder, final List<float[]> vectors) throws IOException {
        // Vector field
        if (dataType == VectorDataType.FLOAT) {
            final float[] vector = SearchTestHelper.generateOneSingleFloatVector(
                DIMENSIONS,
                MIN_VECTOR_ELEMENT_VALUE,
                MAX_VECTOR_ELEMENT_VALUE
            );
            vectors.add(vector);
            builder.startObject().field(KNN_FIELD_NAME, vector).endObject();
        } else if (dataType == VectorDataType.BYTE) {
            final byte[] vector = SearchTestHelper.generateOneSingleByteVector(
                DIMENSIONS,
                MIN_VECTOR_ELEMENT_VALUE,
                MAX_VECTOR_ELEMENT_VALUE
            );
            vectors.add(SearchTestHelper.convertToFloatArray(vector));
            builder.startObject().field(KNN_FIELD_NAME, SearchTestHelper.convertToIntArray(vector)).endObject();
        } else {
            throw new AssertionError();
        }
    }
}
