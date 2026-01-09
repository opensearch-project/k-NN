/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.util.IOUtils;
import org.opensearch.index.fieldvisitor.FieldsVisitor;
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
            if (derivedSourceVectorTransformer.hasFieldsToInject()) {
                delegate.document(docId, new DerivedSourceStoredFieldVisitor(storedFieldVisitor, docId, derivedSourceVectorTransformer));
                return;
            }
        }
        delegate.document(docId, storedFieldVisitor);
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
                derivedSourceReaders.cloneWithMerge(),
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
     * For merging, we need to tell the derived source stored fields reader to skip injecting the source. Otherwise,
     * on merge we will end up just writing the source to disk. We cant override
     * {@link StoredFieldsReader#getMergeInstance()} because it is used elsewhere than just merging.
     *
     * @return Merged instance that wont inject by default
     */
    private StoredFieldsReader cloneForMerge() {
        try {
            return new KNN10010DerivedSourceStoredFieldsReader(
                delegate.getMergeInstance(),
                derivedVectorFields,
                derivedSourceReaders.cloneWithMerge(),
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
