/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.generate;

import lombok.RequiredArgsConstructor;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.List;

/**
 * This util class is to generate random documents based on {@link IndexingType}.
 * Each document will have below fields based in {@link IndexingType}
 * - Non Nested
 *   - id_field: term
 *   - filter_field: term
 *   - knn_field: KNN Vector, 128 dimension.
 * <p>
 * - Nested
 *   - nested_field: nested, will have 3 nested child documents.
 *   - id_field: term
 *   - filter_field: term
 *   - knn_field: KNN Vector
 * <p>
 *
 * For sparse case, it will make only 80% document have KNN field. As a result, 20% of document will not have knn field.
 * Also for testing filtering functionality, filter_field will have either filter-0 or filter-1. Therefore, applying filter will cut off
 * half of documents. See {@link Documents#validateResponse(List)}
 */
@RequiredArgsConstructor
public abstract class DocumentsGenerator {
    public static final String NESTED_FIELD_NAME = "nested_field_name";
    public static final String KNN_FIELD_NAME = "knn_field";
    public static final String FILTER_FIELD_NAME = "filter_field";
    public static final String ID_FIELD_NAME = "id_field";
    public static final float MIN_VECTOR_ELEMENT_VALUE = -100;
    public static final float MAX_VECTOR_ELEMENT_VALUE = 100;
    public static final int DIMENSIONS = 128;
    public static final int NUM_CHILD_DOCS = 3;
    public static final float DENSE_RATIO = 0.8f;
    public static final int FILTER_ID_NO_MOD = 2;

    protected final int numDocuments;
    protected final IndexingType indexingType;
    protected final VectorDataType dataType;

    public static DocumentsGenerator create(final IndexingType indexingType, final VectorDataType dataType, final int numDocuments) {

        if (indexingType.isNested()) {
            return new NestedDocumentsGenerator(numDocuments, indexingType, dataType);
        }

        return new NonNestedDocumentsGenerator(numDocuments, indexingType, dataType);
    }

    public Documents generate() {
        final List<String> documents = new ArrayList<>(numDocuments);
        final List<List<float[]>> vectors = new ArrayList<>();
        for (int i = 0; i < numDocuments; ++i) {
            final GeneratedDocAndVector docAndVector = generateOneDoc(i);
            documents.add(docAndVector.docString);
            vectors.add(docAndVector.vectors);
        }

        return new Documents(documents, vectors);
    }

    protected abstract GeneratedDocAndVector generateOneDoc(int idNo);

    protected record GeneratedDocAndVector(String docString, List<float[]> vectors) {
    }
}
