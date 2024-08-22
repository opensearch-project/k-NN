/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes quantization states to off heap memory
 */
public class KNNQuantizationStateWriter {

    private final IndexOutput output;
    private List<FieldQuantizationState> fieldQuantizationStates = new ArrayList<>();

    /**
     * Constructor
     * @param segmentWriteState segment write state containing segment information
     * @throws IOException exception could be thrown while creating the output
     */
    public KNNQuantizationStateWriter(SegmentWriteState segmentWriteState) throws IOException {
        String quantizationStateFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX
        );

        output = segmentWriteState.directory.createOutput(quantizationStateFileName, segmentWriteState.context);
    }

    /**
     * Writes an index header
     * @param segmentWriteState state containing segment information
     * @throws IOException exception could be thrown while writing header
     */
    public void writeHeader(SegmentWriteState segmentWriteState) throws IOException {
        CodecUtil.writeIndexHeader(output, "QuantizationCodec", 0, segmentWriteState.segmentInfo.getId(), segmentWriteState.segmentSuffix);
    }

    /**
     * Writes a quantization state as bytes
     * @param fieldName field name
     * @param quantizationState quantization state
     * @throws IOException could be thrown while writing
     */
    public void writeState(String fieldName, QuantizationState quantizationState) throws IOException {
        byte[] stateBytes = quantizationState.toByteArray();
        long position = output.getFilePointer();
        output.writeBytes(stateBytes, stateBytes.length);
        fieldQuantizationStates.add(new FieldQuantizationState(fieldName, stateBytes, position));
    }

    /**
     * Writes index footer and other index information for parsing later
     * @throws IOException could be thrown while writing
     */
    public void writeFooter() throws IOException {
        long indexStartPosition = output.getFilePointer();
        output.writeInt(fieldQuantizationStates.size());
        for (FieldQuantizationState fieldQuantizationState : fieldQuantizationStates) {
            output.writeString(fieldQuantizationState.fieldName);
            output.writeInt(fieldQuantizationState.stateBytes.length);
            output.writeVLong(fieldQuantizationState.position);
        }
        output.writeLong(indexStartPosition);
        output.writeInt(-1);
        CodecUtil.writeFooter(output);
        output.close();
        fieldQuantizationStates = new ArrayList<>();
    }

    @AllArgsConstructor
    private static class FieldQuantizationState {
        final String fieldName;
        final byte[] stateBytes;
        final Long position;
    }
}