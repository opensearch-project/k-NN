/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.util.IOUtils;
import org.opensearch.index.fieldvisitor.FieldsVisitor;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec.KNN9120DerivedSourceStoredFieldsFormat.DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY;
import static org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec.KNN9120DerivedSourceStoredFieldsFormat.DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE;

public class KNN9120DerivedSourceStoredFieldsReader extends StoredFieldsReader {
    private final StoredFieldsReader delegate;
    private final List<FieldInfo> derivedVectorFields;
    private final KNN9120DerivedSourceReadersSupplier derivedSourceReadersSupplier;
    private final SegmentReadState segmentReadState;
    private final boolean shouldInject;

    private final DerivedSourceVectorInjector derivedSourceVectorInjector;

    /**
     *
     * @param delegate delegate StoredFieldsReader
     * @param derivedVectorFields List of fields that are derived source fields
     * @param derivedSourceReadersSupplier Supplier for the derived source readers
     * @param segmentReadState SegmentReadState for the segment
     * @throws IOException in case of I/O error
     */
    public KNN9120DerivedSourceStoredFieldsReader(
        StoredFieldsReader delegate,
        List<FieldInfo> derivedVectorFields,
        KNN9120DerivedSourceReadersSupplier derivedSourceReadersSupplier,
        SegmentReadState segmentReadState
    ) throws IOException {
        this(delegate, derivedVectorFields, derivedSourceReadersSupplier, segmentReadState, true);
    }

    private KNN9120DerivedSourceStoredFieldsReader(
        StoredFieldsReader delegate,
        List<FieldInfo> derivedVectorFields,
        KNN9120DerivedSourceReadersSupplier derivedSourceReadersSupplier,
        SegmentReadState segmentReadState,
        boolean shouldInject
    ) throws IOException {
        this.delegate = delegate;
        this.derivedVectorFields = derivedVectorFields;
        this.derivedSourceReadersSupplier = derivedSourceReadersSupplier;
        this.segmentReadState = segmentReadState;
        this.shouldInject = shouldInject;
        this.derivedSourceVectorInjector = createDerivedSourceVectorInjector();
    }

    private DerivedSourceVectorInjector createDerivedSourceVectorInjector() throws IOException {
        return new DerivedSourceVectorInjector(derivedSourceReadersSupplier, segmentReadState, derivedVectorFields);
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
            delegate.document(docId, new KNN9120DerivedSourceStoredFieldVisitor(storedFieldVisitor, docId, derivedSourceVectorInjector));
            return;
        }
        delegate.document(docId, storedFieldVisitor);
    }

    @Override
    public StoredFieldsReader clone() {
        try {
            return new KNN9120DerivedSourceStoredFieldsReader(
                delegate.clone(),
                derivedVectorFields,
                derivedSourceReadersSupplier,
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
        IOUtils.close(delegate, derivedSourceVectorInjector);
    }

    /**
     * Checks if any of the segments being merged contains legacy segments. If so, we need to use the legacy codec
     * for merging.
     *
     * @param mergeState {@link MergeState}
     * @return true if any of the segments being merged contains legacy segments, false otherwise
     */
    public static boolean doesMergeContainLegacySegments(MergeState mergeState) {
        for (int i = 0; i < mergeState.storedFieldsReaders.length; i++) {
            if (mergeState.storedFieldsReaders[i] instanceof KNN9120DerivedSourceStoredFieldsReader
                && doesSegmentContainLegacyFields(mergeState.fieldInfos[i])) {
                return true;
            }
        }
        return false;
    }

    private static boolean doesSegmentContainLegacyFields(FieldInfos fieldInfos) {
        for (FieldInfo fieldInfo : fieldInfos) {
            if (DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE.equals(fieldInfo.attributes().get(DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY))) {
                return true;
            }
        }
        return false;
    }
}
