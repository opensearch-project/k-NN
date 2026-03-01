/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Getter;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.Nullable;

import java.io.Closeable;
import java.io.IOException;

/**
 * Holds the Lucene readers required to reconstruct vector fields for derived source.
 *
 * <p>Derived source allows OpenSearch to reconstruct the original {@code _source} document from
 * stored field data rather than persisting raw source bytes. This class bundles the two readers
 * needed for that reconstruction:
 * <ul>
 *   <li>{@link KnnVectorsReader} - reads raw vector values from the segment.</li>
 *   <li>{@link DocValuesProducer} - reads doc-values (e.g. quantized or nested vector data).
 *   This is needed for backward compatibility for indices created before 2.17 </li>
 * </ul>
 *
 * <p>Either reader may be {@code null} when the corresponding field type is absent in a segment;
 * callers must null-check before use.
 *
 * <p><b>Lifecycle:</b> The owning instance (created via the public constructor) is responsible for
 * closing both underlying readers. Cloned instances (via {@link #clone()}) and merge instances
 * (via {@link #getMergeInstance()}) are non-owning: their {@link #close()} is a no-op, so only
 * the original instance drives resource cleanup.
 */
@Getter
public final class DerivedSourceReaders implements Cloneable, Closeable {
    @Nullable
    private final KnnVectorsReader knnVectorsReader;
    @Nullable
    private final DocValuesProducer docValuesProducer;
    private final Closeable onClose;

    /**
     * Creates an owning instance. Closing this instance will close both underlying readers.
     *
     * @param knnVectorsReader  reader for raw vector values; may be {@code null}.
     * @param docValuesProducer reader for doc-values; may be {@code null}.
     */
    public DerivedSourceReaders(KnnVectorsReader knnVectorsReader, DocValuesProducer docValuesProducer) {
        assert knnVectorsReader != null || docValuesProducer != null : "At least one reader must be non-null";
        this.knnVectorsReader = knnVectorsReader;
        this.docValuesProducer = docValuesProducer;
        this.onClose = () -> IOUtils.closeWhileHandlingException(knnVectorsReader, docValuesProducer);
    }

    private DerivedSourceReaders(KnnVectorsReader knnVectorsReader, DocValuesProducer docValuesProducer, Closeable onClose) {
        assert knnVectorsReader != null || docValuesProducer != null : "At least one reader must be non-null";
        this.knnVectorsReader = knnVectorsReader;
        this.docValuesProducer = docValuesProducer;
        this.onClose = onClose;
    }

    /**
     * Returns a non-owning view of this instance for use during Lucene segment merges.
     * The returned instance's {@link #close()} is a no-op; the original instance retains
     * ownership of the underlying readers. See
     * <a href="https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L3372">IndexWriter</a>
     * for context on why merging does not close readers.
     *
     * {@link #clone()} and {@link #getMergeInstance()} are kept separate to avoid any side effects between the two
     * if the behavior ever changes
     *
     * @return a non-owning {@code DerivedSourceReaders} sharing the same underlying readers.
     */
    public DerivedSourceReaders getMergeInstance() {
        return new DerivedSourceReaders(knnVectorsReader, docValuesProducer, () -> {});
    }

    /**
     * Returns a non-owning shallow clone sharing the same underlying readers.
     * The cloned instance's {@link #close()} is a no-op.
     *
     * {@link #clone()} and {@link #getMergeInstance()} are kept separate to avoid any side effects between the two
     * if the behavior ever changes
     *
     * @return a non-owning {@code DerivedSourceReaders} sharing the same underlying readers.
     */
    @Override
    public DerivedSourceReaders clone() {
        return new DerivedSourceReaders(knnVectorsReader, docValuesProducer, () -> {});
    }

    /**
     * Closes the underlying readers if this is an owning instance. No-op for cloned or merge instances.
     */
    @Override
    public void close() throws IOException {
        onClose.close();
    }
}
