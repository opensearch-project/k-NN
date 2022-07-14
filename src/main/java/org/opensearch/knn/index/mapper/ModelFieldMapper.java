/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Field mapper for model in mapping
 */
public class ModelFieldMapper extends KNNVectorFieldMapper {

    ModelFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        ModelDao modelDao,
        String modelId
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo, ignoreMalformed, stored, hasDocValues);

        this.modelId = modelId;
        this.modelDao = modelDao;

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.putAttribute(MODEL_ID, modelId);
        this.fieldType.freeze();
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        // For the model field mapper, we cannot validate the model during index creation due to
        // an issue with reading cluster state during mapper creation. So, we need to validate the
        // model when ingestion starts.
        ModelMetadata modelMetadata = this.modelDao.getMetadata(modelId);

        if (modelMetadata == null) {
            throw new IllegalStateException(
                "Model \""
                    + modelId
                    + "\" from "
                    + context.mapperService().index().getName()
                    + "'s mapping does not exist. Because the "
                    + "\""
                    + MODEL_ID
                    + "\" parameter is not updateable, this index will need to "
                    + "be recreated with a valid model."
            );
        }

        parseCreateField(context, modelMetadata.getDimension());
    }
}
