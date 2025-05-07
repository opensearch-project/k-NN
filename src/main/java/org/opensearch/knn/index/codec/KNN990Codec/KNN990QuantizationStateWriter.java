/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.AllArgsConstructor;
import lombok.Setter;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.profiler.SegmentProfilerState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes quantization states to off heap memory
 */
public final class KNN990QuantizationStateWriter {

    private final IndexOutput output;
    private List<FieldQuantizationState> fieldQuantizationStates = new ArrayList<>();
    static final String NATIVE_ENGINES_990_KNN_VECTORS_FORMAT_QS_DATA = "NativeEngines990KnnVectorsFormatQSData";

    /**
     * Constructor
     * Overall file format for writer:
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
     * @param segmentWriteState segment write state containing segment information
     * @throws IOException exception could be thrown while creating the output
     */
    public KNN990QuantizationStateWriter(SegmentWriteState segmentWriteState, String fileSuffix) throws IOException {
        String stateFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            fileSuffix
        );

        output = segmentWriteState.directory.createOutput(stateFileName, segmentWriteState.context);
    }

    public KNN990QuantizationStateWriter(SegmentWriteState segmentWriteState) throws IOException {
        this(segmentWriteState, KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX);
    }

    /**
     * Writes an index header
     * @param segmentWriteState state containing segment information
     * @throws IOException exception could be thrown while writing header
     */
    public void writeHeader(SegmentWriteState segmentWriteState) throws IOException {
        CodecUtil.writeIndexHeader(
            output,
            NATIVE_ENGINES_990_KNN_VECTORS_FORMAT_QS_DATA,
            0,
            segmentWriteState.segmentInfo.getId(),
            segmentWriteState.segmentSuffix
        );
    }

    /**
     * Writes a quantization state as bytes
     *
     * @param fieldNumber field number
     * @param quantizationState quantization state
     * @throws IOException could be thrown while writing
     */
    public void writeState(int fieldNumber, QuantizationState quantizationState) throws IOException {
        byte[] stateBytes = quantizationState.toByteArray();
        long position = output.getFilePointer();
        output.writeBytes(stateBytes, stateBytes.length);
        fieldQuantizationStates.add(new FieldQuantizationState(fieldNumber, stateBytes, position));
    }

    /**
     * Writes a segment profile state as bytes
     *
     * @param fieldNumber field number
     * @param segmentProfilerState segment profiler state
     * @throws IOException could be thrown while writing
     */
    public void writeState(int fieldNumber, SegmentProfilerState segmentProfilerState) throws IOException {
        byte[] stateBytes = segmentProfilerState.toByteArray();
        long position = output.getFilePointer();
        output.writeBytes(stateBytes, stateBytes.length);
        fieldQuantizationStates.add(new FieldQuantizationState(fieldNumber, stateBytes, position));
    }

    /**
     * Writes index footer and other index information for parsing later
     * @throws IOException could be thrown while writing
     */
    public void writeFooter() throws IOException {
        long indexStartPosition = output.getFilePointer();
        output.writeInt(fieldQuantizationStates.size());
        for (FieldQuantizationState fieldQuantizationState : fieldQuantizationStates) {
            output.writeInt(fieldQuantizationState.fieldNumber);
            output.writeInt(fieldQuantizationState.stateBytes.length);
            output.writeVLong(fieldQuantizationState.position);
        }
        output.writeLong(indexStartPosition);
        output.writeInt(-1);
        CodecUtil.writeFooter(output);
    }

    @AllArgsConstructor
    private static class FieldQuantizationState {
        final int fieldNumber;
        final byte[] stateBytes;
        @Setter
        Long position;
    }

    public void closeOutput() throws IOException {
        output.close();
    }
}
