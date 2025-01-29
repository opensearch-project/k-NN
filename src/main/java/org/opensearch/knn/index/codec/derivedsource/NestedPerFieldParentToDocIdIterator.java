/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Iterator over the children documents of a particular parent
 */
public class NestedPerFieldParentToDocIdIterator {

    private final FieldInfo childFieldInfo;
    private final SegmentReadState segmentReadState;
    private final DerivedSourceReaders derivedSourceReaders;
    private final int parentDocId;
    private final int previousParentDocId;
    private final List<Integer> children;
    private int currentChild;

    /**
     *
     * @param childFieldInfo FieldInfo for the child field
     * @param segmentReadState SegmentReadState for the segment
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @param parentDocId Parent docId of the parent
     * @throws IOException if there is an error reading the parent docId
     */
    public NestedPerFieldParentToDocIdIterator(
        FieldInfo childFieldInfo,
        SegmentReadState segmentReadState,
        DerivedSourceReaders derivedSourceReaders,
        int parentDocId
    ) throws IOException {
        this.childFieldInfo = childFieldInfo;
        this.segmentReadState = segmentReadState;
        this.derivedSourceReaders = derivedSourceReaders;
        this.parentDocId = parentDocId;
        this.previousParentDocId = previousParent();
        this.children = getChildren();
        this.currentChild = -1;
    }

    /**
     * For the given parent get its first child offset
     *
     * @return the first child offset. If there are no children, just return NO_MORE_DOCS
     */
    public int firstChild() {
        if (parentDocId - previousParentDocId == 1) {
            return NO_MORE_DOCS;
        }
        return previousParentDocId + 1;
    }

    /**
     * Get the next child for this parent
     *
     * @return the next child docId. If this has not been set, return -1. If there are no more children, return
     * NO_MORE_DOCS
     */
    public int nextChild() {
        currentChild++;
        if (currentChild >= children.size()) {
            return NO_MORE_DOCS;
        }
        return children.get(currentChild);
    }

    /**
     * Get the current child for this parent
     *
     * @return the current child docId. If this has not been set, return -1
     */
    public int childId() {
        return children.get(currentChild);
    }

    /**
     *
     * @return the number of children for this parent
     */
    public int numChildren() {
        return children.size();
    }

    /**
     * For parentDocId of this class, find the one just before it to be used for matching children.
     *
     * @return the parent docId just before the parentDocId. -1 if none exist
     * @throws IOException if there is an error reading the parent docId
     */
    private int previousParent() throws IOException {
        // TODO: In the future this needs to be generalized to handle multiple levels of nesting
        // For now, for non-nested docs, the primary_term field can be used to identify root level docs. For reference:
        // https://github.com/opensearch-project/OpenSearch/blob/2.18.0/server/src/main/java/org/opensearch/search/fetch/subphase/SeqNoPrimaryTermPhase.java#L72
        // https://github.com/opensearch-project/OpenSearch/blob/3032bef54d502836789ea438f464ae0b1ba978b2/server/src/main/java/org/opensearch/index/mapper/SeqNoFieldMapper.java#L206-L230
        // We use it here to identify the previous parent to the current parent to get a range on the children documents
        FieldInfo seqTermsFieldInfo = segmentReadState.fieldInfos.fieldInfo("_primary_term");
        NumericDocValues numericDocValues = derivedSourceReaders.getDocValuesProducer().getNumeric(seqTermsFieldInfo);
        int previousParentDocId = -1;
        while (numericDocValues.nextDoc() != NO_MORE_DOCS) {
            if (numericDocValues.docID() >= parentDocId) {
                break;
            }
            previousParentDocId = numericDocValues.docID();
        }
        return previousParentDocId;
    }

    /**
     * Get all the children that match the parent path for the _nested_field
     *
     * @return list of children that match the parent path
     * @throws IOException if there is an error reading the children
     */
    private List<Integer> getChildren() throws IOException {
        if (this.parentDocId - this.previousParentDocId <= 1) {
            return Collections.emptyList();
        }

        // First, we need to get the currect PostingsEnum for the key as _nested_path and the value the actual parent
        // path.
        String childField = childFieldInfo.name;
        String parentField = ParentChildHelper.getParentField(childField);

        Terms terms = derivedSourceReaders.getFieldsProducer().terms("_nested_path");
        if (terms == null) {
            return Collections.emptyList();
        }
        TermsEnum nestedFieldsTerms = terms.iterator();
        BytesRef childPathRef = new BytesRef(parentField);
        PostingsEnum postingsEnum = null;
        while (nestedFieldsTerms.next() != null) {
            BytesRef currentTerm = nestedFieldsTerms.term();
            if (currentTerm.bytesEquals(childPathRef)) {
                postingsEnum = nestedFieldsTerms.postings(null);
                break;
            }
        }

        // Next, get all the children that match this parent path. If none were found, return an empty list
        if (postingsEnum == null) {
            return Collections.emptyList();
        }
        List<Integer> children = new ArrayList<>();
        postingsEnum.advance(previousParentDocId + 1);
        while (postingsEnum.docID() != NO_MORE_DOCS && postingsEnum.docID() < parentDocId) {
            if (postingsEnum.freq() > 0) {
                children.add(postingsEnum.docID());
            }
            postingsEnum.nextDoc();
        }

        return children;
    }
}
