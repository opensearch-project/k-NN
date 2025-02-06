/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.util.IOUtils;
import org.opensearch.index.fieldvisitor.FieldsVisitor;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReaders;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceStoredFieldVisitor;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceVectorInjector;

import java.io.IOException;
import java.util.List;

@Log4j2
public class DerivedSourceStoredFieldsReader extends StoredFieldsReader {
    private final StoredFieldsReader delegate;
    private final List<FieldInfo> derivedVectorFields;
    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;
    private final boolean shouldInject;

    private final DerivedSourceVectorInjector derivedSourceVectorInjector;

    /**
     *
     * @param delegate delegate StoredFieldsReader
     * @param derivedVectorFields List of fields that are derived source fields
     * @param derivedSourceReaders Derived source readers
     * @param segmentReadState SegmentReadState for the segment
     * @throws IOException in case of I/O error
     */
    public DerivedSourceStoredFieldsReader(
        StoredFieldsReader delegate,
        List<FieldInfo> derivedVectorFields,
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState
    ) throws IOException {
        this(delegate, derivedVectorFields, derivedSourceReaders, segmentReadState, true);
    }

    private DerivedSourceStoredFieldsReader(
        StoredFieldsReader delegate,
        List<FieldInfo> derivedVectorFields,
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState,
        boolean shouldInject
    ) throws IOException {
        this.delegate = delegate;
        this.derivedVectorFields = derivedVectorFields;
        this.derivedSourceReaders = derivedSourceReaders;
        this.segmentReadState = segmentReadState;
        this.shouldInject = shouldInject;
        this.derivedSourceVectorInjector = createDerivedSourceVectorInjector();
    }

    private DerivedSourceVectorInjector createDerivedSourceVectorInjector() {
        return new DerivedSourceVectorInjector(derivedSourceReaders, segmentReadState, derivedVectorFields);
    }

    @Override
    public void document(int docId, StoredFieldVisitor storedFieldVisitor) throws IOException {
        // If the visitor has explicitly indicated it does not need the fields, we should not inject them
        boolean isVisitorNeedFields = true;
        if (storedFieldVisitor instanceof FieldsVisitor) {
            isVisitorNeedFields = derivedSourceVectorInjector.shouldInject(
                ((FieldsVisitor) storedFieldVisitor).includes(),
                ((FieldsVisitor) storedFieldVisitor).excludes()
            );
        }
        if (shouldInject && isVisitorNeedFields) {
            delegate.document(docId, new DerivedSourceStoredFieldVisitor(storedFieldVisitor, docId, derivedSourceVectorInjector));
            return;
        }
        delegate.document(docId, storedFieldVisitor);
    }

    @Override
    public StoredFieldsReader clone() {
        try {
            return new DerivedSourceStoredFieldsReader(
                delegate.clone(),
                derivedVectorFields,
                derivedSourceReaders.clone(),
                segmentReadState,
                shouldInject
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        log.debug("Closing derived source stored fields reader for segment: " + segmentReadState.segmentInfo.name);
        IOUtils.close(delegate, derivedSourceVectorInjector);
    }

    /**
     * For merging, we need to tell the derived source stored fields reader to skip injecting the source. Otherwise,
     * on merge we will end up just writing the source to disk. We cant override
     * {@link StoredFieldsReader#getMergeInstance()} because it is used elsewhere than just merging.
     *
     * @return Merged instance that wont inject by default
     */
    private StoredFieldsReader cloneForMerge() {
        try {
            return new DerivedSourceStoredFieldsReader(
                delegate.getMergeInstance(),
                derivedVectorFields,
                derivedSourceReaders.clone(),
                segmentReadState,
                false
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * For merging, we need to tell the derived source stored fields reader to skip injecting the source. Otherwise,
     * on merge we will end up just writing the source to disk
     *
     * @param storedFieldsReader stored fields reader to wrap
     * @return wrapped stored fields reader
     */
    public static StoredFieldsReader wrapForMerge(StoredFieldsReader storedFieldsReader) {
        if (storedFieldsReader instanceof DerivedSourceStoredFieldsReader) {
            return ((DerivedSourceStoredFieldsReader) storedFieldsReader).cloneForMerge();
        }
        return storedFieldsReader;
    }
}
