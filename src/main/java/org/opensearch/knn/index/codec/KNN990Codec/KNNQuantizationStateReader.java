/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateReadConfig;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads quantization states
 */
public final class KNNQuantizationStateReader {

    /**
     * Read quantization states and return list of fieldNames and bytes
     * File format:
     * Header
     * QS1 state bytes
     * QS2 state bytes
     * Number of quantization states
     * QS1 field number
     * QS1 state bytes length
     * QS1 position of state bytes
     * QS2 field number
     * QS2 state bytes length
     * QS2 position of state bytes
     * Position of index section (where QS1 field name is located)
     * -1 (marker)
     * Footer
     *
     * @param state the read state to read from
     */
    public static Map<String, byte[]> read(SegmentReadState state) throws IOException {
        String quantizationStateFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX
        );
        Map<String, byte[]> readQuantizationStateInfos = new HashMap<>();

        try (IndexInput input = state.directory.openInput(quantizationStateFileName, IOContext.READ)) {
            CodecUtil.retrieveChecksum(input);

            int numFields = getNumFields(input);

            List<Integer> fieldNumbers = new ArrayList<>();
            List<Long> positions = new ArrayList<>();
            List<Integer> lengths = new ArrayList<>();

            // Read each field's metadata from the index section
            for (int i = 0; i < numFields; i++) {
                fieldNumbers.add(input.readInt());
                int length = input.readInt();
                lengths.add(length);
                long position = input.readVLong();
                positions.add(position);
            }
            // Read each field's bytes
            for (int i = 0; i < numFields; i++) {
                input.seek(positions.get(i));
                byte[] stateBytes = new byte[lengths.get(i)];
                input.readBytes(stateBytes, 0, lengths.get(i));
                String fieldName = state.fieldInfos.fieldInfo(fieldNumbers.get(i)).getName();
                readQuantizationStateInfos.put(fieldName, stateBytes);
            }
        }
        return readQuantizationStateInfos;
    }

    /**
     * Reads an individual quantization state for a given field
     * @param readConfig a config class that contains necessary information for reading the state
     * @return quantization state
     */
    public static QuantizationState read(QuantizationStateReadConfig readConfig) throws IOException {
        String quantizationStateFileName = IndexFileNames.segmentFileName(
            readConfig.getSegmentName(),
            readConfig.getSegmentSuffix(),
            KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX
        );
        int fieldNumber = readConfig.getFieldInfo().getFieldNumber();

        try (IndexInput input = readConfig.getDirectory().openInput(quantizationStateFileName, IOContext.READ)) {
            CodecUtil.retrieveChecksum(input);
            int numFields = getNumFields(input);

            long position = -1;
            int length = 0;

            // Read each field's metadata from the index section, break when correct field is found
            for (int i = 0; i < numFields; i++) {
                int tempFieldNumber = input.readInt();
                int tempLength = input.readInt();
                long tempPosition = input.readVLong();
                if (tempFieldNumber == fieldNumber) {
                    position = tempPosition;
                    length = tempLength;
                    break;
                }
            }

            if (position == -1 || length == 0) {
                throw new IllegalArgumentException(String.format("Field %s not found", readConfig.getFieldInfo().getName()));
            }

            input.seek(position);
            byte[] stateBytes = new byte[length];
            input.readBytes(stateBytes, 0, length);
            // Deserialize the byte array to a quantization state object
            ScalarQuantizationType scalarQuantizationType = ScalarQuantizationType.fromId(
                Integer.parseInt(readConfig.getScalarQuantizationTypeId())
            );
            if (scalarQuantizationType == ScalarQuantizationType.ONE_BIT) {
                return OneBitScalarQuantizationState.fromByteArray(stateBytes);
            } else if (scalarQuantizationType == ScalarQuantizationType.TWO_BIT
                || scalarQuantizationType == ScalarQuantizationType.FOUR_BIT) {
                    return MultiBitScalarQuantizationState.fromByteArray(stateBytes);
                } else {
                    throw new IllegalArgumentException(String.format("Unexpected scalar quantization type: %s", scalarQuantizationType));
                }
        }
    }

    @VisibleForTesting
    static int getNumFields(IndexInput input) throws IOException {
        long footerStart = input.length() - CodecUtil.footerLength();
        long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
        input.seek(markerAndIndexPosition);
        long indexStartPosition = input.readLong();
        input.readInt();
        input.seek(indexStartPosition);
        return input.readInt();
    }
}
