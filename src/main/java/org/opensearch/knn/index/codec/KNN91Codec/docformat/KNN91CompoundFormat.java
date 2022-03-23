/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.KNN91Codec.docformat;

import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Class to encode/decode compound file
 */
public class KNN91CompoundFormat extends CompoundFormat {

    private final CompoundFormat delegate;

    /**
     * Constructor that takes a delegate to handle non-overridden methods
     *
     * @param delegate CompoundFormat that will handle non-overridden methods
     */
    public KNN91CompoundFormat(CompoundFormat delegate) {
        this.delegate = delegate;
    }

    @Override
    public CompoundDirectory getCompoundReader(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        return delegate.getCompoundReader(dir, si, context);
    }

    @Override
    public void write(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        for (KNNEngine knnEngine : KNNEngine.values()) {
            writeEngineFiles(dir, si, context, knnEngine.getExtension());
        }
        delegate.write(dir, si, context);
    }

    private void writeEngineFiles(Directory dir, SegmentInfo si, IOContext context, String engineExtension) throws IOException {
        /*
         * If engine file present, remove it from the compounding file list to avoid header/footer checks
         * and create a new compounding file format with extension engine + c.
         */
        Set<String> engineFiles = si.files().stream().filter(file -> file.endsWith(engineExtension)).collect(Collectors.toSet());

        Set<String> segmentFiles = new HashSet<>(si.files());

        if (!engineFiles.isEmpty()) {
            for (String engineFile : engineFiles) {
                String engineCompoundFile = engineFile + KNNConstants.COMPOUND_EXTENSION;
                dir.copyFrom(dir, engineFile, engineCompoundFile, context);
            }
            segmentFiles.removeAll(engineFiles);
            si.setFiles(segmentFiles);
        }
    }
}
