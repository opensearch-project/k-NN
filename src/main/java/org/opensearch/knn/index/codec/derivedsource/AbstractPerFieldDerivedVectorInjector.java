/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

@Log4j2
abstract class AbstractPerFieldDerivedVectorInjector implements PerFieldDerivedVectorInjector {
    /**
     * Utility method for formatting the vector values based on the vector data type. KNNVectorValues must be advanced
     * to the correct position.
     *
     * @param fieldInfo fieldinfo for the vector field
     * @param vectorValues vector values of the field. getVector or getConditionalVector should return expected vector.
     * @return vector formatted based on the vector data type
     * @throws IOException if unable to deserialize stored vector
     */
    protected Object formatVector(FieldInfo fieldInfo, KNNVectorValues<?> vectorValues) throws IOException {
        Object vectorValue = vectorValues.getVector();
        // If the vector value is a byte[], we must deserialize
        if (vectorValue instanceof byte[]) {
            BytesRef vectorBytesRef = new BytesRef((byte[]) vectorValue);
            VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
            return KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
        }
        return vectorValues.conditionalCloneVector();
    }
}
