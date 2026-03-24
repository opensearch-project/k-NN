/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNVectorUtil;
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
        return formatVector(fieldInfo, vectorSupplier, vectorCloneSupplier, () -> 1.0f);
    }

    /**
     * Utility method for formatting the vector values based on the vector data type, with lazy denormalization.
     * The norm is only read from doc values when the vector actually needs denormalization (Faiss engine),
     * avoiding unnecessary I/O for Lucene engine vectors.
     *
     * @param fieldInfo fieldinfo for the vector field
     * @param vectorSupplier supplies vector (without clone)
     * @param vectorCloneSupplier supplies clone of vector.
     * @param normSupplier lazily supplies the L2 norm. Only invoked when denormalization is needed.
     * @return vector formatted based on the vector data type. Typically, this will be a float[] or int[].
     * @throws IOException if unable to deserialize stored vector
     */
    protected Object formatVector(
        FieldInfo fieldInfo,
        CheckedSupplier<Object, IOException> vectorSupplier,
        CheckedSupplier<Object, IOException> vectorCloneSupplier,
        CheckedSupplier<Float, IOException> normSupplier
    ) throws IOException {
        Object vectorValue = vectorSupplier.get();
        // If the vector value is a byte[], we must deserialize.
        // This is the Faiss/nmslib path where vectors are stored as BinaryDocValues.
        if (vectorValue instanceof byte[]) {
            BytesRef vectorBytesRef = new BytesRef((byte[]) vectorValue);
            VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
            Object deserialized = KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
            if (deserialized instanceof float[] floatVector) {
                float norm = normSupplier.get();
                if (norm != 1.0f) {
                    KNNVectorUtil.denormalize(floatVector, norm, true);
                }
            }
            return deserialized;
        }
        float[] vector = (float[]) vectorCloneSupplier.get();
        float norm = normSupplier.get();
        if (norm != 1.0f) {
            KNNVectorUtil.denormalize(vector, norm, true);
        }
        return vector;
    }
}
