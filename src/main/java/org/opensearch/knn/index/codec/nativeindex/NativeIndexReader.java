/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.isSegmentUsingLegacyIndexCompoundCodec;

import java.io.Closeable;
import java.io.IOException;

import lombok.extern.log4j.Log4j2;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

@Log4j2
public class NativeIndexReader implements Closeable {

    private final SegmentInfo segmentInfo;
    private final Directory directory;

    public NativeIndexReader(final SegmentInfo segmentInfo) throws IOException {
        this.segmentInfo = segmentInfo;
        if (!segmentInfo.getUseCompoundFile() || isSegmentUsingLegacyIndexCompoundCodec(segmentInfo)) {
            directory = segmentInfo.dir;
        } else {
            // compound directory opens the .cfs file that needs to be closed with directory
            directory = segmentInfo.getCodec().compoundFormat().getCompoundReader(segmentInfo.dir, segmentInfo);
        }
    }

    public static NativeIndexReader getReader(SegmentInfo segmentInfo) throws IOException {
        return new NativeIndexReader(segmentInfo);
    }

    public IndexInput open(String indexFileName) throws IOException {
        IndexInput input = directory.openInput(indexFileName, IOContext.READONCE);

        // If not using legacy codec, we need to strip header and footer after checking the header
        if (!isSegmentUsingLegacyIndexCompoundCodec(segmentInfo)) {
            // Validates the header
            CodecUtil.checkIndexHeader(
                input,
                NativeIndexWriter.NATIVE_INDEX_CODEC_NAME,
                NativeIndexWriter.NATIVE_INDEX_CODEC_MIN_VERSION,
                NativeIndexWriter.NATIVE_INDEX_CODEC_CURRENT_VERSION,
                segmentInfo.getId(),
                NativeIndexWriter.NATIVE_INDEX_CODEC_SUFFIX
            );
            long headerLength = CodecUtil.indexHeaderLength(
                NativeIndexWriter.NATIVE_INDEX_CODEC_NAME, NativeIndexWriter.NATIVE_INDEX_CODEC_SUFFIX
            );
            log.info("[KNN] returning sliced input");

            return new NativeIndexInput(
                "CompoundNativeIndex",
                input,
                input.slice("testSlice", headerLength, calculateIndexSize(indexFileName))
            );
        }

        return input;
    }

    public long calculateIndexSize(String indexFileName) throws IOException {
        if (isSegmentUsingLegacyIndexCompoundCodec(segmentInfo)) {
            return directory.fileLength(indexFileName);
        }

        // If it is non-legacy compound file, calculate size from file size - header - footer
        // Cache will hold the index file bytes after the header hence the size calculation
        long compoundFileLength = directory.fileLength(indexFileName);
        long headerLength = CodecUtil.indexHeaderLength(NativeIndexWriter.NATIVE_INDEX_CODEC_NAME, NativeIndexWriter.NATIVE_INDEX_CODEC_SUFFIX);
        return compoundFileLength - headerLength - CodecUtil.footerLength();
    }

    @Override
    public void close() throws IOException {
        // We don't need to close the directory if it is same as segment info dir.
        // If we created any other directory like compound format directory, we need to close this resource.
        if (directory != segmentInfo.dir) {
            directory.close();
        }
    }


}
