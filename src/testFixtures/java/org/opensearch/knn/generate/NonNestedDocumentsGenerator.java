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
 * Non-nested document JSON strings generator.
 */
public class NonNestedDocumentsGenerator extends DocumentsGenerator {
    public NonNestedDocumentsGenerator(final int numDocuments, final IndexingType indexingType, final VectorDataType dataType) {
        super(numDocuments, indexingType, dataType);
    }

    protected GeneratedDocAndVector generateOneDoc(final int idNo) {
        try {
            XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
            final List<float[]> vectors = new ArrayList<>();

            // Id field
            final String id = "id-" + idNo;
            builder.field(ID_FIELD_NAME, id);

            // Filtering field
            final String filter = "filter-" + (idNo % FILTER_ID_NO_MOD);
            builder.field(FILTER_FIELD_NAME, filter);

            if (indexingType == IndexingType.DENSE || ThreadLocalRandom.current().nextFloat() < DENSE_RATIO) {
                // Vector field
                if (dataType == VectorDataType.FLOAT) {
                    final float[] vector = SearchTestHelper.generateOneSingleFloatVector(
                        DIMENSIONS,
                        MIN_VECTOR_ELEMENT_VALUE,
                        MAX_VECTOR_ELEMENT_VALUE,
                        false
                    );
                    vectors.add(vector);
                    builder.field(KNN_FIELD_NAME, vector);
                } else if (dataType == VectorDataType.BYTE) {
                    final byte[] vector = SearchTestHelper.generateOneSingleByteVector(
                        DIMENSIONS,
                        MIN_VECTOR_ELEMENT_VALUE,
                        MAX_VECTOR_ELEMENT_VALUE
                    );
                    vectors.add(SearchTestHelper.convertToFloatArray(vector));
                    builder.field(KNN_FIELD_NAME, SearchTestHelper.convertToIntArray(vector));
                } else if (dataType == VectorDataType.BINARY) {
                    final byte[] binaryVector = SearchTestHelper.generateOneSingleBinaryVector(DIMENSIONS);
                    vectors.add(SearchTestHelper.convertToFloatArray(binaryVector));
                    builder.field(KNN_FIELD_NAME, SearchTestHelper.convertToIntArray(binaryVector));
                } else {
                    throw new AssertionError();
                }
            }

            final String docString = builder.endObject().toString();
            return new GeneratedDocAndVector(docString, vectors);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
