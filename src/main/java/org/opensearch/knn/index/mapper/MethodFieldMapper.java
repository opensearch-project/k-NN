/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.common.Explicit;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_SIGNED_BYTE_SQ;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

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

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);

        this.fieldType.putAttribute(DIMENSION, String.valueOf(dimension));
        this.fieldType.putAttribute(SPACE_TYPE, knnMethodContext.getSpaceType().getValue());
        this.fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());

        KNNEngine knnEngine = knnMethodContext.getKnnEngine();
        this.fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());

        try {
            Map<String, Object> libParams = knnEngine.getKNNLibraryIndexingContext(knnMethodContext).getLibraryParameters();

            // If VectorDataType is Byte using Faiss engine then manipulate Index Description to use "SQ8_direct_signed" scalar quantizer
            // For example, Index Description "HNSW16,Flat" will be updated as "HNSW16,SQ8_direct_signed"
            if (VectorDataType.BYTE.equals(vectorDataType) && libParams.containsKey(INDEX_DESCRIPTION_PARAMETER)) {
                String indexDescriptionValue = (String) libParams.get(INDEX_DESCRIPTION_PARAMETER);
                String updatedIndexDescription = indexDescriptionValue.split(",")[0] + "," + FAISS_SIGNED_BYTE_SQ;
                libParams.replace(INDEX_DESCRIPTION_PARAMETER, updatedIndexDescription);
            }
            this.fieldType.putAttribute(PARAMETERS, XContentFactory.jsonBuilder().map(libParams).toString());
        } catch (IOException ioe) {
            throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe));
        }

        this.fieldType.freeze();
    }
}
