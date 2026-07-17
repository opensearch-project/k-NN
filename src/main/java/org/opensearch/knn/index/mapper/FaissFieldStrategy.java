/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.engine.faiss.SQConfig;
import org.opensearch.knn.index.engine.faiss.SQConfigParser;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Faiss (and NMSLIB) engine implementation of {@link EngineFieldStrategy}.
 * Handles field type construction and vector field creation for non-Lucene KNN engines.
 * Vector field creation delegates to the parent class default behavior via returning null
 * from createFloatFields/createByteFields, signaling the caller to use its default path.
 */
public final class FaissFieldStrategy implements EngineFieldStrategy {

    public static final FaissFieldStrategy INSTANCE = new FaissFieldStrategy();

    private FaissFieldStrategy() {}

    @Override
    public FieldTypeConfig buildFieldTypeConfig(
        KNNMappingConfig knnMappingConfig,
        KNNMethodContext resolvedKnnMethodContext,
        KNNLibraryIndexingContext knnLibraryIndexingContext,
        VectorDataType vectorDataType,
        Version indexCreatedVersion,
        boolean hasDocValues
    ) {
        boolean useLuceneBasedVectorField = KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(indexCreatedVersion);
        KNNEngine knnEngine = resolvedKnnMethodContext.getKnnEngine();
        QuantizationConfig quantizationConfig = knnLibraryIndexingContext.getQuantizationConfig();

        FieldType fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        fieldType.putAttribute(DIMENSION, String.valueOf(knnMappingConfig.getDimension()));
        fieldType.putAttribute(SPACE_TYPE, resolvedKnnMethodContext.getSpaceType().getValue());

        ResolvedIndexSpec spec = knnLibraryIndexingContext.getResolvedSpec();
        // 1-bit quantization has its own per-field format that handles quantization internally,
        // so we set sq_config instead of qframe_config
        if (spec != null && spec.isSQOneBit()) {
            SQConfig sqConfig = SQConfig.builder().bits(spec.getQuantizationBits().getValue()).build();
            fieldType.putAttribute(SQ_CONFIG, SQConfigParser.toCsv(sqConfig));
        } else {
            if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
                fieldType.putAttribute(QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(quantizationConfig));
            }
        }

        fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());
        try {
            fieldType.putAttribute(
                PARAMETERS,
                XContentFactory.jsonBuilder().map(knnLibraryIndexingContext.getLibraryParameters()).toString()
            );
        } catch (IOException ioe) {
            throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe), ioe);
        }

        if (useLuceneBasedVectorField) {
            int adjustedDimension = vectorDataType == VectorDataType.BINARY
                ? knnMappingConfig.getDimension() / 8
                : knnMappingConfig.getDimension();
            final VectorEncoding encoding = vectorDataType == VectorDataType.FLOAT ? VectorEncoding.FLOAT32 : VectorEncoding.BYTE;
            final VectorSimilarityFunction similarityFunction = findBestMatchingVectorSimilarityFunction(
                resolvedKnnMethodContext.getSpaceType(),
                indexCreatedVersion
            );
            fieldType.setVectorAttributes(adjustedDimension, encoding, similarityFunction);
        } else {
            fieldType.setDocValuesType(DocValuesType.BINARY);
        }

        fieldType.freeze();

        VectorTransformer vectorTransformer = knnLibraryIndexingContext.getVectorTransformer();
        return new FieldTypeConfig(fieldType, null, vectorTransformer, useLuceneBasedVectorField);
    }

    @Override
    public List<Field> createFloatFields(
        String name,
        float[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        return null;
    }

    @Override
    public List<Field> createByteFields(
        String name,
        byte[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        return null;
    }

    private static VectorSimilarityFunction findBestMatchingVectorSimilarityFunction(SpaceType spaceType, Version indexCreatedVersion) {
        if (indexCreatedVersion.onOrAfter(Version.V_3_0_0)) {
            try {
                return spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
            } catch (Exception e) {
                // SpaceType may not have a direct Lucene VectorSimilarityFunction mapping (e.g. HAMMING, L1).
                // Fall back to DEFAULT similarity function for backward compatibility with pre-3.0 behavior.
            }
        }
        return SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
    }
}
