/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;

import java.io.IOException;

public abstract class AbstractPerFieldDerivedVectorTransformer implements PerFieldDerivedVectorTransformer {
    /**
     * Utility method for formatting the vector values based on the vector data type.
     *
     * @param fieldInfo fieldinfo for the vector field
     * @param vectorSupplier supplies vector (without clone)
     * @param vectorCloneSupplier supplies clone of vector.
     * @return vector formatted based on the vector data type. Typically, this will be a float[] or int[].
     * @throws IOException if unable to deserialize stored vector
     */
    protected Object formatVector(
        FieldInfo fieldInfo,
        CheckedSupplier<Object, IOException> vectorSupplier,
        CheckedSupplier<Object, IOException> vectorCloneSupplier
    ) throws IOException {
        return formatVector(fieldInfo, vectorSupplier, vectorCloneSupplier, 1.0f);
    }

    /**
     * Utility method for formatting the vector values based on the vector data type, with optional denormalization.
     *
     * @param fieldInfo fieldinfo for the vector field
     * @param vectorSupplier supplies vector (without clone)
     * @param vectorCloneSupplier supplies clone of vector.
     * @param norm L2 norm to apply for denormalization. 1.0f means no denormalization.
     * @return vector formatted based on the vector data type. Typically, this will be a float[] or int[].
     * @throws IOException if unable to deserialize stored vector
     */
    protected Object formatVector(
        FieldInfo fieldInfo,
        CheckedSupplier<Object, IOException> vectorSupplier,
        CheckedSupplier<Object, IOException> vectorCloneSupplier,
        float norm
    ) throws IOException {
        Object vectorValue = vectorSupplier.get();
        // If the vector value is a byte[], we must deserialize
        if (vectorValue instanceof byte[]) {
            BytesRef vectorBytesRef = new BytesRef((byte[]) vectorValue);
            VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
            Object deserialized = KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
            if (norm != 1.0f && deserialized instanceof float[] floatVector) {
                denormalize(floatVector, norm);
            }
            return deserialized;
        }
        float[] vector = (float[]) vectorCloneSupplier.get();
        if (norm != 1.0f) {
            denormalize(vector, norm);
        }
        return vector;
    }

    private static void denormalize(float[] vector, float norm) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= norm;
        }
    }
}
