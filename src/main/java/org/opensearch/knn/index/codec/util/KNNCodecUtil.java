/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import lombok.NonNull;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentInfo;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80BinaryDocValues;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

public class KNNCodecUtil {
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;

    /**
     * This method provides a rough estimate of the number of bytes used for storing an array with the given parameters.
     * @param numVectors number of vectors in the array
     * @param vectorLength the length of each vector
     * @param vectorDataType type of data stored in each vector
     * @return rough estimate of number of bytes used to store an array with the given parameters
     */
    public static long calculateArraySize(int numVectors, int vectorLength, VectorDataType vectorDataType) {
        if (vectorDataType == VectorDataType.FLOAT) {
            return numVectors * vectorLength * FLOAT_BYTE_SIZE;
        } else if (vectorDataType == VectorDataType.BINARY || vectorDataType == VectorDataType.BYTE) {
            return numVectors * vectorLength;
        } else {
            throw new IllegalArgumentException(
                "Float, binary, and byte are the only supported vector data types for array size calculation."
            );
        }
    }

    public static String buildEngineFileName(String segmentName, String latestBuildVersion, String fieldName, String extension) {
        return String.format("%s%s%s", buildEngineFilePrefix(segmentName), latestBuildVersion, buildEngineFileSuffix(fieldName, extension));
    }

    public static String buildEngineFilePrefix(String segmentName) {
        return String.format("%s_", segmentName);
    }

    public static String buildEngineFileSuffix(String fieldName, String extension) {
        return String.format("_%s%s", fieldName, extension);
    }

    public static long getTotalLiveDocsCount(final BinaryDocValues binaryDocValues) {
        long totalLiveDocs;
        if (binaryDocValues instanceof KNN80BinaryDocValues) {
            totalLiveDocs = ((KNN80BinaryDocValues) binaryDocValues).getTotalLiveDocs();
        } else {
            totalLiveDocs = binaryDocValues.cost();
        }
        return totalLiveDocs;
    }

    /**
     * Get Engine Files from segment with specific fieldName and engine extension
     *
     * @param extension Engine extension comes from {@link KNNEngine#getExtension()}}
     * @param fieldName Filed for knn field
     * @param segmentInfo {@link SegmentInfo} One Segment info to use for compute.
     * @return List of engine files
     */
    public static List<String> getEngineFiles(String extension, String fieldName, SegmentInfo segmentInfo) {
        /*
         * In case of compound file, extension would be <engine-extension> + c otherwise <engine-extension>
         */
        String engineExtension = segmentInfo.getUseCompoundFile() ? extension + KNNConstants.COMPOUND_EXTENSION : extension;
        String engineSuffix = fieldName + engineExtension;
        String underLineEngineSuffix = "_" + engineSuffix;

        List<String> engineFiles = segmentInfo.files()
            .stream()
            .filter(fileName -> fileName.endsWith(underLineEngineSuffix))
            .sorted(Comparator.comparingInt(String::length))
            .collect(Collectors.toList());
        return engineFiles;
    }

    /**
     * Get engine file name from given field and segment info.
     * Ex: _0_165_my_field.faiss
     *
     * @param field : Field info that might have a vector index file. Not always it has it.
     * @param segmentInfo : Segment where we are collecting an engine file list.
     * @return : Found vector engine names, if not found, returns null.
     */
    public static String getNativeEngineFileFromFieldInfo(FieldInfo field, SegmentInfo segmentInfo) {
        if (!field.attributes().containsKey(KNN_FIELD)) {
            return null;
        }
        // Only Native Engine put into indexPathMap
        final KNNEngine knnEngine = getNativeKNNEngine(field);
        if (knnEngine == null) {
            return null;
        }
        final List<String> engineFiles = KNNCodecUtil.getEngineFiles(knnEngine.getExtension(), field.name, segmentInfo);
        if (engineFiles.isEmpty()) {
            return null;
        } else {
            final String vectorIndexFileName = engineFiles.get(0);
            return vectorIndexFileName;
        }
    }

    /**
     * Get KNNEngine From FieldInfo
     *
     * @param field which field we need produce from engine
     * @return if and only if Native Engine we return specific engine, else return null
     */
    private static KNNEngine getNativeKNNEngine(@NonNull FieldInfo field) {
        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(field);
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(engine)) {
            return engine;
        }
        return null;
    }
}
