/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.MapperService;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opensearch.knn.common.FieldInfoExtractor.isMemoryOptimizedSearchField;

/**
 * Handles warmup for k-NN fields that use memory-optimized (disk-based) search.
 * <p>
 * Memory-optimized search keeps vector data on disk rather than loading it entirely into
 * the JVM heap. Warming up these fields forces the underlying data into the OS page cache
 * so that the first real searches avoid cold-read latency.
 * <p>
 * The warmup is performed by issuing a no-op search via the codec's
 * {@link org.apache.lucene.codecs.KnnVectorsReader#search} with a {@code null} query vector,
 * which triggers the reader to touch the on-disk structures without producing results.
 */
@Log4j2
public class MemoryOptimizedSearchWarmup {
    /**
     * Warms up all memory-optimized k-NN fields in the given leaf reader.
     *
     * @param leafReader    the segment-level reader to warm up
     * @param mapperService the mapper service used to resolve field types; if {@code null}, no warmup is performed
     * @param indexName     the name of the index, used to check memory-optimized search support
     * @return a list of field names that were successfully warmed up
     */
    public List<String> warmUp(final LeafReader leafReader, final MapperService mapperService, final String indexName) {
        if (mapperService == null) {
            return Collections.emptyList();
        }

        final SegmentReader segmentReader = Lucene.segmentReader(leafReader);

        final List<FieldInfo> memOptSearchFields = getFieldsForMemoryOptimizedSearch(leafReader, mapperService, indexName);
        final List<String> warmedUp = new ArrayList<>();

        for (FieldInfo field : memOptSearchFields) {
            if (warmUpField(field, segmentReader)) {
                warmedUp.add(field.getName());
            }
        }

        return warmedUp;
    }

    /**
     * Warms up a single k-NN field by issuing a no-op search through the codec's vector reader.
     *
     * @param field         the field to warm up
     * @param segmentReader the segment reader providing access to the vector reader
     * @return {@code true} if the warmup succeeded, {@code false} if an exception occurred
     */
    private boolean warmUpField(final FieldInfo field, final SegmentReader segmentReader) {
        try {
            assert segmentReader.getVectorReader() instanceof PerFieldKnnVectorsFormat.FieldsReader : "Expected PerFieldKnnVectorsFormat"
                + ".FieldsReader";
            final KnnVectorsReader vectorsReader = ((PerFieldKnnVectorsFormat.FieldsReader) segmentReader.getVectorReader()).getFieldReader(
                field.getName()
            );
            assert vectorsReader instanceof WarmableReader;
            if (vectorsReader instanceof WarmableReader warmableReader) {
                log.info("Warming up reader for field: {}", field.getName());
                warmableReader.warmUp(field.getName());
            }
            return true;
        } catch (Exception e) {
            // Expected during warmup initialization
            log.error("Warm up failed for {}", field.getName(), e);
            return false;
        }
    }

    /**
     * Collects all {@link FieldInfo} entries in the segment that are eligible for memory-optimized search warmup.
     *
     * @param leafReader    the segment-level reader
     * @param mapperService the mapper service for resolving field types
     * @param indexName     the index name
     * @return a list of fields that support memory-optimized search
     */
    private List<FieldInfo> getFieldsForMemoryOptimizedSearch(
        final LeafReader leafReader,
        final MapperService mapperService,
        String indexName
    ) {
        final List<FieldInfo> fields = new ArrayList<>();
        for (FieldInfo field : leafReader.getFieldInfos()) {
            if (isMemoryOptimizedSearchField(field, mapperService, indexName)) {
                fields.add(field);
            }
        }
        return fields;
    }
}
