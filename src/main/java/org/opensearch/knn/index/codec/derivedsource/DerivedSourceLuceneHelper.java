/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.mapper.FieldNamesFieldMapper;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Utility class used to implement Lucene functionality that can be used to interact with Lucene
 */
@RequiredArgsConstructor
public class DerivedSourceLuceneHelper {

    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;

    /**
     * Return list of documents matching term in range
     *
     * @param startDocId first eligible document (inclusive)
     * @param endDocId last eligible document (inclusive)
     * @param termField field to check for term
     * @param term term to match
     * @return list of docIds that match the term in the given range
     * @throws IOException if there is an issue reading
     */
    public List<Integer> termMatchesInRange(int startDocId, int endDocId, String termField, String term) throws IOException {
        if (endDocId - startDocId < 0) {
            return Collections.emptyList();
        }

        // First, we need to get the current PostingsEnum for the key as term field and term
        PostingsEnum postingsEnum = getPostingsEnum(termField, term);

        // Next, get all the docs that match this parent path. If none were found, return an empty list
        if (postingsEnum == null) {
            return Collections.emptyList();
        }
        List<Integer> matchingDocs = new ArrayList<>();
        postingsEnum.advance(startDocId);
        while (postingsEnum.docID() != NO_MORE_DOCS && postingsEnum.docID() <= endDocId) {
            if (postingsEnum.freq() > 0) {
                matchingDocs.add(postingsEnum.docID());
            }
            postingsEnum.nextDoc();
        }

        return matchingDocs;
    }

    /**
     * Check if the docId is a parent for the given field. To do this, it checks if any of the documents in the range
     * contain the parent field in the _nested_path
     *
     * @param offset First doc to check (inclusive)
     * @param parentDocId document to be checked if its a parent
     * @return true if the docId is a parent, false otherwise
     */
    public boolean isNestedParent(int offset, int parentDocId, FieldInfo childFieldInfo) throws IOException {
        if (parentDocId <= 0) {
            return false;
        }
        KNNVectorValues<?> vectorValues = KNNVectorValuesFactory.getVectorValues(
            childFieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
        if (vectorValues == null) {
            return false;
        }
        return vectorValues.advance(offset) < parentDocId;
    }

    /**
     * Given a document id, get its parent docId.
     *
     * @param docId DocId to map to parent
     * @param limit Last eligible doc (inclusive)
     * @param parentFieldName Field to check
     * @return Doc Id of first matching doc on or after offset that contains the parent field name. {@link DocIdSetIterator#NO_MORE_DOCS} if not found
     * @throws IOException if unable to read segment
     */
    public int getParentDocId(int docId, int limit, String parentFieldName) throws IOException {
        List<Integer> matches = termMatchesInRange(docId, limit, "_nested_path", parentFieldName);
        if (matches.isEmpty()) {
            return NO_MORE_DOCS;
        }
        return matches.getFirst();
    }

    /**
     * Get the lowest docId for a field that is greater than (or equal to) the offset. This method is implemented in a
     * very similar way as checking if a field exists.
     *
     * @param fieldToMatch field to find the lowest docId for
     * @param offset       offset to start searching from (inclusive)
     * @return lowest docId for the field that is greater than the offset. Returns {@link DocIdSetIterator#NO_MORE_DOCS} if doc cannot be found
     * @throws IOException if there is an issue reading from the formats
     */
    public int getFirstDocWhereFieldExists(String fieldToMatch, int offset) throws IOException {
        // This method implementation is inspired by the FieldExistsQuery in Lucene and the FieldNamesMapper in
        // Opensearch. We first mimic the logic in the FieldExistsQuery in order to identify the docId of the nested
        // doc. If that fails, we rely on
        // References:
        // 1. https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/FieldExistsQuery.java#L170-L218.
        // 2.
        // https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/index/mapper/FieldMapper.java#L316-L324
        FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(fieldToMatch);

        if (fieldInfo == null) {
            return NO_MORE_DOCS;
        }

        DocIdSetIterator iterator = null;
        if (fieldInfo.hasNorms() && derivedSourceReaders.getNormsProducer() != null) { // the field indexes norms
            iterator = derivedSourceReaders.getNormsProducer().getNorms(fieldInfo);
        } else if (fieldInfo.getVectorDimension() != 0 && derivedSourceReaders.getKnnVectorsReader() != null) { // the field indexes vectors
            switch (fieldInfo.getVectorEncoding()) {
                case FLOAT32:
                    iterator = derivedSourceReaders.getKnnVectorsReader().getFloatVectorValues(fieldInfo.name).iterator();
                    break;
                case BYTE:
                    iterator = derivedSourceReaders.getKnnVectorsReader().getByteVectorValues(fieldInfo.name).iterator();
                    break;
            }
        } else if (fieldInfo.getDocValuesType() != DocValuesType.NONE && derivedSourceReaders.getDocValuesProducer() != null) { // the field
            // indexes
            // doc
            // values
            switch (fieldInfo.getDocValuesType()) {
                case NUMERIC:
                    iterator = derivedSourceReaders.getDocValuesProducer().getNumeric(fieldInfo);
                    break;
                case BINARY:
                    iterator = derivedSourceReaders.getDocValuesProducer().getBinary(fieldInfo);
                    break;
                case SORTED:
                    iterator = derivedSourceReaders.getDocValuesProducer().getSorted(fieldInfo);
                    break;
                case SORTED_NUMERIC:
                    iterator = derivedSourceReaders.getDocValuesProducer().getSortedNumeric(fieldInfo);
                    break;
                case SORTED_SET:
                    iterator = derivedSourceReaders.getDocValuesProducer().getSortedSet(fieldInfo);
                    break;
                case NONE:
                default:
                    throw new AssertionError();
            }
        }
        if (iterator != null) {
            return iterator.advance(offset);
        }

        // Check the field names field type for matches
        PostingsEnum postingsEnum = getPostingsEnum(FieldNamesFieldMapper.NAME, fieldInfo.name);
        if (postingsEnum == null) {
            return NO_MORE_DOCS;
        }
        return postingsEnum.advance(offset);
    }

    /**
     * Get the first child of the given parentDoc. This can be used to determine if the document contains any nested
     * fields.
     *
     * @return doc id of last matching doc. {@link DocIdSetIterator#NO_MORE_DOCS} if no children exist.
     * @throws IOException
     */
    public int getFirstChild(int parentDocId) throws IOException {
        // If its the first document id, then there is no change there are parents
        if (parentDocId == 0) {
            return NO_MORE_DOCS;
        }

        // Only root level documents have the "_primary_term" field. So, we iterate through all of the documents in
        // order to find out if any have this term.
        // TODO: This is expensive and should be optimized. We should start at doc parentDocId - 10000 and work back
        // (can we fetch the setting? Maybe)
        FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo("_primary_term");
        assert derivedSourceReaders.getDocValuesProducer() != null;
        NumericDocValues numericDocValues = derivedSourceReaders.getDocValuesProducer().getNumeric(fieldInfo);
        int previousParentDocId = NO_MORE_DOCS;
        numericDocValues.advance(0);
        while (numericDocValues.docID() != NO_MORE_DOCS) {
            if (numericDocValues.docID() >= parentDocId) {
                break;
            }
            previousParentDocId = numericDocValues.docID();
            numericDocValues.nextDoc();
        }

        // If there are no numeric docvalues before the current parent doc, then the parent doc is the first parent. So
        // its first child must be 0
        if (previousParentDocId == NO_MORE_DOCS) {
            return 0;
        }
        // If the document right before is the previous parent, then there are no children.
        if (parentDocId - previousParentDocId <= 1) {
            return NO_MORE_DOCS;
        }
        return previousParentDocId + 1;
    }

    private PostingsEnum getPostingsEnum(String termField, String term) throws IOException {
        if (derivedSourceReaders.getFieldsProducer() == null) {
            return null;
        }
        Terms terms = derivedSourceReaders.getFieldsProducer().terms(termField);
        if (terms == null) {
            return null;
        }
        TermsEnum nestedFieldsTerms = terms.iterator();
        BytesRef childPathRef = new BytesRef(term);
        PostingsEnum postingsEnum = null;
        while (nestedFieldsTerms.next() != null) {
            BytesRef currentTerm = nestedFieldsTerms.term();
            if (currentTerm.bytesEquals(childPathRef)) {
                postingsEnum = nestedFieldsTerms.postings(null);
                break;
            }
        }
        return postingsEnum;
    }
}
