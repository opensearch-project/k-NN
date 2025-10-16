/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;

/**
 * Fully warm up the index by loading every byte from disk, causing page faults
 */
@Log4j2
public class FullFieldWarmUpStrategy extends FieldWarmUpStrategy {
    private final SegmentReader segmentReader;
    private final Directory directory;

    public FullFieldWarmUpStrategy(LeafReader leafReader, Directory directory) {
        this.segmentReader = Lucene.segmentReader(leafReader);
        this.directory = directory;
    }

    private void warmUpFile(String file) throws IOException {
        try (IndexInput input = directory.openInput(file, IOContext.READONCE)) {
            for (int i = 0; i < input.length(); i += 4096) {
                input.seek(i);
                input.readByte();
            }
            input.seek(input.length() - 1);
            input.readByte();
        }
    }

    @Override
    public boolean warmUp(FieldInfo field) throws IOException {
        final KNNEngine knnEngine = extractKNNEngine(field);
        final List<String> engineFiles = KNNCodecUtil.getEngineFiles(
            knnEngine.getExtension(),
            field.getName(),
            segmentReader.getSegmentInfo().info
        );
        if (engineFiles.isEmpty()) {
            log.warn("Could not find an engine file for field [{}]", field.getName());
            return false;
        }
        final Path indexPath = Paths.get(engineFiles.getFirst());

        warmUpFile(indexPath.toString());

        return true;
    }
}
