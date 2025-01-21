/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;
import java.nio.file.Path;

public class JVectorFormat extends KnnVectorsFormat {
    public static final String NAME = "JVectorFormat";
    public static final String META_CODEC_NAME = "JVectorVectorsFormatMeta";
    public static final String VECTOR_INDEX_CODEC_NAME = "JVectorVectorsFormatIndex";
    public static final String JVECTOR_FILES_SUFFIX = "jvector";
    public static final String META_EXTENSION = "meta-" + JVECTOR_FILES_SUFFIX;
    public static final String VECTOR_INDEX_EXTENSION = "data-" + JVECTOR_FILES_SUFFIX;
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;
    public static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
    private static final int DEFAULT_MAX_CONN = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;
    private static final float DEFAULT_DEGREE_OVERFLOW = 1.2f;
    private static final float DEFAULT_ALPHA = 1.2f;

    private final int maxConn;
    private final int beamWidth;

    public JVectorFormat() {
        this(NAME, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
    }

    public JVectorFormat(int maxConn, int beamWidth) {
        this(NAME, maxConn, beamWidth);
    }

    public JVectorFormat(String name, int maxConn, int beamWidth) {
        super(name);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
    }



    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new JVectorWriter(state, maxConn, beamWidth, DEFAULT_DEGREE_OVERFLOW, DEFAULT_ALPHA);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new JVectorReader(state);
    }

    static Path getVectorIndexPath(Path directoryBasePath, String baseDataFileName, String field) {
        return directoryBasePath.resolve(baseDataFileName + "_" + field + "." + JVectorFormat.VECTOR_INDEX_EXTENSION);
    }
}
