/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;

import java.io.IOException;

/**
 * This class contains a Lucene's IndexInput with a reader buffer.
 * A Java reference of this class will be passed to native engines, then 'copyBytes' method will be
 * called by native engine via JNI API.
 * Therefore, this class servers as a read layer in native engines to read the bytes it wants.
 */
@Log4j2
public class IndexInputWithBuffer {
    private IndexInput indexInput;
    private long contentLength;
    // 64K buffer.
    private byte[] buffer = new byte[64 * 1024];

    /**
     * Lazily supplies the field's full-precision vectors, used only to reconstruct native flat storage when loading a
     * graph-only .faiss produced by FP32 flat-vector deduplication (IO_FLAG_SKIP_STORAGE). Set right before
     * {@code loadIndex}; the supplier is invoked (once) only if native calls {@link #copyVectors(int)}, which happens
     * only for a graph-only index. For normal (non-deduped) loads it is never invoked, so there is no cost or reader
     * access. Null when reconstruction cannot apply (non-FAISS/non-FP32/quantized).
     */
    @Setter
    private CheckedSupplier<KNNVectorValues<?>, IOException> knnVectorValuesSupplier;
    // Materialized (lazily, on first copyVectors call) from knnVectorValuesSupplier.
    private KNNVectorValues<?> knnVectorValues;
    // Reused staging buffer for batched vector streaming to native (sized maxVectors * dimension, lazily).
    private float[] vectorBuffer;
    // Once the vector iterator is exhausted, do not advance it again: calling nextDoc() past NO_MORE_DOCS on a sparse
    // (IndexedDISI-backed) iterator throws an AssertionError. This lets native call copyVectors one final time safely.
    private boolean vectorsExhausted;

    // Sequential-read state for reconstructing flat storage from .vec. Reconstruction consumes vectors in FAISS-ordinal
    // order, which is exactly the contiguous layout of the Lucene flat file, so it is a pure forward scan of .vec.
    // However the flat reader opens .vec with DataAccessHint.RANDOM (HNSW does random access), which disables OS
    // readahead; without it, streaming the whole file faults the mmap 4KB page by page (hundreds of thousands of
    // synchronous reads on a cold cache). For the one-shot reconstruction scan we flip the flat-vector IndexInput to
    // DataAccessHint.SEQUENTIAL (kernel then reads ahead aggressively), and restore RANDOM once done so later
    // random-access reads (exact search, rescore) are not penalized. This is advisory only (madvise): the bytes
    // returned are unchanged, so the reconstructed IndexFlat is byte-identical and recall/scores do not change.
    // vectorDataSlice is null when the flat reader does not expose its slice, in which case reconstruction proceeds
    // with the original (RANDOM) read path - correct, just not accelerated.
    private boolean sequentialReadsInitAttempted;
    private IndexInput vectorDataSlice;

    public IndexInputWithBuffer(@NonNull IndexInput indexInput) {
        this.indexInput = indexInput;
        this.contentLength = indexInput.length();
    }

    /**
     * This method will be invoked in native engines via JNI API.
     * Then it will call IndexInput to read required bytes then copy them into a read buffer.
     *
     * @param nbytes Desired number of bytes to be read.
     * @return The number of read bytes in a buffer.
     * @throws IOException
     */
    private int copyBytes(long nbytes) throws IOException {
        final int readBytes = (int) Math.min(nbytes, buffer.length);
        indexInput.readBytes(buffer, 0, readBytes);
        return readBytes;
    }

    private long remainingBytes() {
        return contentLength - indexInput.getFilePointer();
    }

    /**
     * Invoked from native engines via JNI to stream full-precision FP32 vectors when reconstructing the native flat
     * storage for a graph-only .faiss (FP32 flat-vector dedup). Advances {@link #knnVectorValues} and stages up to
     * {@code maxVectors} consecutive vectors (row-major, {@code dimension} floats each) into the reused
     * {@link #vectorBuffer}, which native reads under a JNI critical section. Vectors are yielded in doc-id iteration
     * order, which matches the FAISS ordinal order they were added in at write time.
     *
     * @param maxVectors maximum number of vectors to stage in this batch.
     * @return the number of vectors staged (0 once the iterator is exhausted).
     * @throws IOException on read failure.
     */
    private int copyVectors(int maxVectors) throws IOException {
        if (vectorsExhausted) {
            return 0;
        }
        // Materialize the vector values lazily on first use, so normal (non-graph-only) loads never pay for it.
        if (knnVectorValues == null) {
            if (knnVectorValuesSupplier == null) {
                vectorsExhausted = true;
                return 0;
            }
            knnVectorValues = knnVectorValuesSupplier.get();
        }
        // Only full-precision float vectors are reconstructed this way; native never calls this otherwise.
        if ((knnVectorValues instanceof KNNFloatVectorValues) == false) {
            vectorsExhausted = true;
            return 0;
        }
        final KNNFloatVectorValues floatVectorValues = (KNNFloatVectorValues) knnVectorValues;
        // One-time: switch the underlying .vec slice to sequential read advice for this forward scan.
        if (sequentialReadsInitAttempted == false) {
            sequentialReadsInitAttempted = true;
            maybeEnableSequentialReads(floatVectorValues);
        }
        int copied = 0;
        while (copied < maxVectors) {
            if (knnVectorValues.nextDoc() == DocIdSetIterator.NO_MORE_DOCS) {
                vectorsExhausted = true;
                // Scan finished: restore random read advice so subsequent random-access reads are not penalized.
                restoreDefaultReadAdvice();
                break;
            }
            // getVector() returns a shared reference; copy it into the staging buffer.
            final float[] vector = floatVectorValues.getVector();
            final int dimension = vector.length;
            final int requiredCapacity = maxVectors * dimension;
            if (vectorBuffer == null || vectorBuffer.length < requiredCapacity) {
                vectorBuffer = new float[requiredCapacity];
            }
            System.arraycopy(vector, 0, vectorBuffer, copied * dimension, dimension);
            copied++;
        }
        return copied;
    }

    /**
     * Best-effort switch of the Lucene flat-vector {@link IndexInput} (the {@code .vec} slice) backing the field's
     * float vectors to {@link DataAccessHint#SEQUENTIAL}, so the kernel reads ahead aggressively during the forward
     * reconstruction scan instead of faulting page by page. Only the standard flat
     * format, whose values implement {@link HasIndexSlice}, is accelerated; any other shape leaves
     * {@link #vectorDataSlice} null and reconstruction proceeds on the original read path. Never throws: this is an
     * optimization, not a correctness requirement.
     */
    private void maybeEnableSequentialReads(final KNNFloatVectorValues floatVectorValues) {
        try {
            final KNNVectorValuesIterator iterator = floatVectorValues.getVectorValuesIterator();
            if ((iterator instanceof KNNVectorValuesIterator.DocIdsIteratorValues) == false) {
                log.debug("Flat-vector dedup: iterator is not DocIdsIteratorValues; skipping sequential read advice");
                return;
            }
            final KnnVectorValues luceneVectorValues = ((KNNVectorValuesIterator.DocIdsIteratorValues) iterator).getKnnVectorValues();
            if ((luceneVectorValues instanceof HasIndexSlice) == false) {
                log.debug("Flat-vector dedup: values do not expose an index slice; skipping sequential read advice");
                return;
            }
            final IndexInput slice = ((HasIndexSlice) luceneVectorValues).getSlice();
            if (slice == null) {
                return;
            }
            slice.updateIOContext(IOContext.DEFAULT.withHints(DataAccessHint.SEQUENTIAL));
            vectorDataSlice = slice;
            log.debug("Flat-vector dedup: enabled SEQUENTIAL read advice on .vec slice (length={})", slice.length());
        } catch (final Throwable t) {
            // Any failure here just disables the optimization; reconstruction proceeds with the default read path.
            vectorDataSlice = null;
            log.debug("Flat-vector dedup: could not enable sequential read advice; using default read path", t);
        }
    }

    /**
     * Restores {@link DataAccessHint#RANDOM} on the flat-vector slice after the reconstruction scan, so later
     * random-access reads of the same file are not penalized by the sequential hint. Best-effort and idempotent.
     */
    private void restoreDefaultReadAdvice() {
        if (vectorDataSlice == null) {
            return;
        }
        try {
            vectorDataSlice.updateIOContext(IOContext.DEFAULT.withHints(DataAccessHint.RANDOM));
        } catch (final Throwable ignored) {
            // Best-effort restore; nothing actionable if it fails.
        } finally {
            vectorDataSlice = null;
        }
    }

    @Override
    public String toString() {
        return "{indexInput=" + indexInput + ", len(buffer)=" + buffer.length + "}";
    }
}
