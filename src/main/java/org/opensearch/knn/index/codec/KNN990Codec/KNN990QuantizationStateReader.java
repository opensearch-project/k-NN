/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateReadConfig;

import java.io.IOException;

/**
 * Reads quantization states
 */
@Log4j2
public final class KNN990QuantizationStateReader {

    /**
     * Reads an individual quantization state for a given field
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
     * @param readConfig a config class that contains necessary information for reading the state
     * @return quantization state
     */
    public static QuantizationState read(QuantizationStateReadConfig readConfig) throws IOException {
        SegmentReadState segmentReadState = readConfig.getSegmentReadState();
        String field = readConfig.getField();
        String quantizationStateFileName = getQuantizationStateFileName(segmentReadState);
        int fieldNumber = segmentReadState.fieldInfos.fieldInfo(field).getFieldNumber();

        try (IndexInput input = segmentReadState.directory.openInput(quantizationStateFileName, IOContext.DEFAULT)) {

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
                throw new IllegalArgumentException(String.format("Field %s not found", field));
            }

            byte[] stateBytes = readStateBytes(input, position, length);

            // Deserialize the byte array to a quantization state object
            ScalarQuantizationType scalarQuantizationType = ((ScalarQuantizationParams) readConfig.getQuantizationParams()).getSqType();
            switch (scalarQuantizationType) {
                case ONE_BIT:
                    return OneBitScalarQuantizationState.fromByteArray(stateBytes);
                case TWO_BIT:
                case FOUR_BIT:
                    return MultiBitScalarQuantizationState.fromByteArray(stateBytes);
                default:
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
        input.seek(indexStartPosition);
        return input.readInt();
    }

    @VisibleForTesting
    static byte[] readStateBytes(IndexInput input, long position, int length) throws IOException {
        input.seek(position);
        byte[] stateBytes = new byte[length];
        input.readBytes(stateBytes, 0, length);
        return stateBytes;
    }

    @VisibleForTesting
    static String getQuantizationStateFileName(SegmentReadState state) {
        return IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX);
    }
}
