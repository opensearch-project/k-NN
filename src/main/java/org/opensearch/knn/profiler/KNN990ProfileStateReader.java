/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import java.io.IOException;

public final class KNN990ProfileStateReader {

    public static SegmentProfilerState read(SegmentProfileStateReadConfig readConfig) throws IOException {
        SegmentReadState srs = readConfig.getSegmentReadState();
        String field = readConfig.getField();
        String fileName = IndexFileNames.segmentFileName(
            srs.segmentInfo.name,
            srs.segmentSuffix,
            KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX
        );
        int targetField = srs.fieldInfos.fieldInfo(field).getFieldNumber();

        try (IndexInput in = srs.directory.openInput(fileName, IOContext.DEFAULT)) {
            CodecUtil.retrieveChecksum(in);
            int numFields = getNumFields(in);
            long position = -1;
            int length = 0;

            for (int i = 0; i < numFields; i++) {
                int fnum = in.readInt();
                int len = in.readInt();
                long pos = in.readVLong();
                if (fnum == targetField) {
                    position = pos;
                    length = len;
                    break;
                }
            }
            if (position < 0) {
                throw new IllegalArgumentException("Field " + field + " not found in state file");
            }
            in.seek(position);
            byte[] info = new byte[length];
            in.readBytes(info, 0, length);
            return SegmentProfilerState.fromBytes(info);
        }
    }

    @VisibleForTesting
    static int getNumFields(IndexInput in) throws IOException {
        long footerStart = in.length() - CodecUtil.footerLength();
        long markerPos = footerStart - Integer.BYTES - Long.BYTES;
        in.seek(markerPos);
        long indexStart = in.readLong();
        in.seek(indexStart);
        return in.readInt();
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
