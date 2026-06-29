/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.util.IOUtils;
import org.opensearch.index.fieldvisitor.FieldsVisitor;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.knn.index.codec.derivedsource.DerivedFieldInfo;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReaders;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceStoredFieldVisitor;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceVectorTransformer;

import java.io.IOException;
import java.util.List;

public class KNN10010DerivedSourceStoredFieldsReader extends StoredFieldsReader {
    private final StoredFieldsReader delegate;
    private final List<DerivedFieldInfo> derivedVectorFields;
    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;
    private final boolean shouldInject;
    private boolean transformerInitialized = false;

    private final DerivedSourceVectorTransformer derivedSourceVectorTransformer;

    /**
     *
     * @param delegate delegate StoredFieldsReader
     * @param derivedVectorFields List of fields that are derived source fields
     * @param derivedSourceReaders derived source readers
     * @param segmentReadState SegmentReadState for the segment
     * @throws IOException in case of I/O error
     */
    public KNN10010DerivedSourceStoredFieldsReader(
        StoredFieldsReader delegate,
        List<DerivedFieldInfo> derivedVectorFields,
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState
    ) throws IOException {
        this(delegate, derivedVectorFields, derivedSourceReaders, segmentReadState, true);
    }

    private KNN10010DerivedSourceStoredFieldsReader(
        StoredFieldsReader delegate,
        List<DerivedFieldInfo> derivedVectorFields,
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState,
        boolean shouldInject
    ) throws IOException {
        this.delegate = delegate;
        this.derivedVectorFields = derivedVectorFields;
        this.derivedSourceReaders = derivedSourceReaders;
        this.segmentReadState = segmentReadState;
        this.shouldInject = shouldInject;
        this.derivedSourceVectorTransformer = createDerivedSourceVectorTransformer();
    }

    private DerivedSourceVectorTransformer createDerivedSourceVectorTransformer() {
        return new DerivedSourceVectorTransformer(derivedSourceReaders, segmentReadState, derivedVectorFields);
    }

    @Override
    public void document(int docId, StoredFieldVisitor storedFieldVisitor) throws IOException {
        // If the visitor has explicitly indicated it does not need the fields, we should not inject them
        if (shouldInject) {
            initializeTransformerIfNeeded(storedFieldVisitor);
            if (derivedSourceVectorTransformer.hasFieldsToInject() && visitorRequestsSource(storedFieldVisitor)) {
                delegate.document(docId, new DerivedSourceStoredFieldVisitor(storedFieldVisitor, docId, derivedSourceVectorTransformer));
                return;
            }
        }
        delegate.document(docId, storedFieldVisitor);
    }

    /**
     * Derived source injection only rewrites {@code _source}. Scroll-based {@code _delete_by_query} /
     * {@code _update_by_query} fetch metadata fields like {@code _routing} with {@code fetch_source=false},
     * so wrapping those visitors can prevent metadata from being returned.
     */
    private boolean visitorRequestsSource(StoredFieldVisitor storedFieldVisitor) throws IOException {
        if (storedFieldVisitor instanceof FieldsVisitor fieldsVisitor) {
            fieldsVisitor.reset();
            FieldInfo sourceField = segmentReadState.fieldInfos.fieldInfo(SourceFieldMapper.NAME);
            if (sourceField == null) {
                return false;
            }
            StoredFieldVisitor.Status status = fieldsVisitor.needsField(sourceField);
            fieldsVisitor.reset();
            return status == StoredFieldVisitor.Status.YES;
        }
        return true;
    }

    private void initializeTransformerIfNeeded(StoredFieldVisitor visitor) {
        if (transformerInitialized) {
            return;
        }
        String[] includes = null;
        String[] excludes = null;

        if (visitor instanceof FieldsVisitor) {
            includes = ((FieldsVisitor) visitor).includes();
            excludes = ((FieldsVisitor) visitor).excludes();
        }

        derivedSourceVectorTransformer.initialize(includes, excludes);
        transformerInitialized = true;
    }

    @Override
    public StoredFieldsReader clone() {
        try {
            return new KNN10010DerivedSourceStoredFieldsReader(
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
        IOUtils.close(delegate, derivedSourceReaders);
    }

    /**
     * Scroll and other sequential stored-field fetches use {@link StoredFieldsReader#getMergeInstance()}.
     * Segment merges explicitly call {@link #wrapForMerge(StoredFieldsReader)} to disable injection.
     */
    @Override
    public StoredFieldsReader getMergeInstance() {
        try {
            return new KNN10010DerivedSourceStoredFieldsReader(
                delegate.getMergeInstance(),
                derivedVectorFields,
                derivedSourceReaders.getMergeInstance(),
                segmentReadState,
                true
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * For merging, we need to tell the derived source stored fields reader to skip injecting the source. Otherwise,
     * on merge we will end up just writing the source to disk.
     *
     * @return Merged instance that wont inject by default
     */
    private StoredFieldsReader cloneForMerge() {
        try {
            return new KNN10010DerivedSourceStoredFieldsReader(
                delegate.getMergeInstance(),
                derivedVectorFields,
                derivedSourceReaders.getMergeInstance(),
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
        if (storedFieldsReader instanceof KNN10010DerivedSourceStoredFieldsReader) {
            return ((KNN10010DerivedSourceStoredFieldsReader) storedFieldsReader).cloneForMerge();
        }
        return storedFieldsReader;
    }

    /**
     * Returns the list of derived vector fields for this reader.
     * Used during merge to collect field names from source segments.
     */
    public List<DerivedFieldInfo> getDerivedVectorFields() {
        return derivedVectorFields;
    }

}
