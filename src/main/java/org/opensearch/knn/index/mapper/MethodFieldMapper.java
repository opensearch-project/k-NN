/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * Field mapper for method definition in mapping
 */
public class MethodFieldMapper extends KNNVectorFieldMapper {

    MethodFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        KNNMethodContext knnMethodContext
    ) {

        super(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodContext.getMethodComponentContext().getIndexVersion()
        );

        this.knnMethod = knnMethodContext;

        this.fieldType = new FieldType();
        this.fieldType.setTokenized(false);
        this.fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNN_FIELD, "true"); // This attribute helps to determine knn field type

        this.fieldType.putAttribute(DIMENSION, String.valueOf(dimension));
        this.fieldType.putAttribute(SPACE_TYPE, knnMethodContext.getSpaceType().getValue());

        KNNEngine knnEngine = knnMethodContext.getKnnEngine();
        this.fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());

        // This for new VectorValuesFormat only enabling it for Faiss right now. We will change this to a version check later on .
        if (knnMethodContext.getMethodComponentContext().getIndexVersion().before(Version.V_2_15_0)) {
            // fieldType.setVectorAttributes(dimension, VectorEncoding.FLOAT32,
            // knnMethodContext.getSpaceType().getVectorSimilarityFunction());
            // } else {
            fieldType.setDocValuesType(DocValuesType.BINARY);
        }

        try {
            this.fieldType.putAttribute(
                PARAMETERS,
                XContentFactory.jsonBuilder().map(knnEngine.getMethodAsMap(knnMethodContext)).toString()
            );
        } catch (IOException ioe) {
            throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe));
        }

        this.fieldType.freeze();
    }
}
