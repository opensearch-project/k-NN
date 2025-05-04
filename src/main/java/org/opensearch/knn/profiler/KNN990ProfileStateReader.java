/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;

/**
 * Reader class for segment profiler states
 */
@Log4j2
public final class KNN990ProfileStateReader {

    /**
     * Reads a segment profiler state for a given field
     *
     * @param readConfig config for reading the profiler state
     * @return SegmentProfilerState object
     * @throws IOException if there's an error reading the state
     */
    public static SegmentProfilerState read(SegmentProfileStateReadConfig readConfig) throws IOException {
        SegmentReadState segmentReadState = readConfig.getSegmentReadState();
        String field = readConfig.getField();
        String stateFileName = getProfileStateFileName(segmentReadState);
        int fieldNumber = segmentReadState.fieldInfos.fieldInfo(field).getFieldNumber();

        try (IndexInput input = segmentReadState.directory.openInput(stateFileName, IOContext.DEFAULT)) {
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
            return SegmentProfilerState.fromBytes(stateBytes);
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
    static String getProfileStateFileName(SegmentReadState state) {
        return IndexFileNames.segmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                KNNConstants.SEGMENT_PROFILE_STATE_FILE_SUFFIX
        );
    }
}
