/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

@Log4j2
@AllArgsConstructor
public class NestedPerFieldDerivedVectorInjector extends AbstractPerFieldDerivedVectorInjector {

    private final FieldInfo childFieldInfo;
    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;

    @Override
    public void inject(int parentDocId, Map<String, Object> sourceAsMap) throws IOException {
        // If the parent has the field, then it is just an object field.
        int lowestDocIdForFieldWithParentAsOffset = getLowestDocIdForField(childFieldInfo.name, parentDocId);
        if (lowestDocIdForFieldWithParentAsOffset == parentDocId) {
            injectObject(parentDocId, sourceAsMap);
            return;
        }

        // Setup the iterator. Return if no parent
        String childFieldName = ParentChildHelper.getChildField(childFieldInfo.name);
        String parentFieldName = ParentChildHelper.getParentField(childFieldInfo.name);
        if (parentFieldName == null) {
            return;
        }
        NestedPerFieldParentToDocIdIterator nestedPerFieldParentToDocIdIterator = new NestedPerFieldParentToDocIdIterator(
            childFieldInfo,
            segmentReadState,
            derivedSourceReaders,
            parentDocId
        );

        if (nestedPerFieldParentToDocIdIterator.numChildren() == 0) {
            return;
        }

        // Initializes the parent field so that there is a list to put each of the children
        Object originalParentValue = sourceAsMap.get(parentFieldName);
        List<Map<String, Object>> reconstructedSource;
        if (originalParentValue instanceof Map) {
            reconstructedSource = new ArrayList<>(List.of((Map<String, Object>) originalParentValue));
        } else {
            reconstructedSource = (List<Map<String, Object>>) originalParentValue;
        }

        // Contains the docIds of existing objects in the map in order. This is used to help figure out the best play
        // to put back the vectors
        List<Integer> positions = mapObjectsToPositionInNestedList(
            reconstructedSource,
            nestedPerFieldParentToDocIdIterator.firstChild(),
            parentDocId
        );

        // Finally, inject children for the document into the source. This code is non-trivial because filtering out
        // the vectors during write could mean that children docs disappear from the source. So, to properly put
        // everything back, we need to figure out where the existing fields in the original map to
        KNNVectorValues<?> vectorValues = KNNVectorValuesFactory.getVectorValues(
            childFieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
        int offsetPositionsIndex = 0;
        while (nestedPerFieldParentToDocIdIterator.nextChild() != NO_MORE_DOCS) {
            // If the child does not have a vector, vectValues advance will advance past child to the next matching
            // docId. So, we need to ensure that doing this does not pass the parent docId.
            if (nestedPerFieldParentToDocIdIterator.childId() > vectorValues.docId()) {
                vectorValues.advance(nestedPerFieldParentToDocIdIterator.childId());
            }
            if (vectorValues.docId() != nestedPerFieldParentToDocIdIterator.childId()) {
                continue;
            }

            int docId = nestedPerFieldParentToDocIdIterator.childId();
            boolean isInsert = true;
            int position = positions.size(); // by default we insert it at the end
            for (int i = offsetPositionsIndex; i < positions.size(); i++) {
                if (docId < positions.get(i)) {
                    position = i;
                    break;
                }
                if (docId == positions.get(i)) {
                    isInsert = false;
                    position = i;
                    break;
                }
            }

            if (isInsert) {
                reconstructedSource.add(position, new HashMap<>());
                positions.add(position, docId);
            }
            reconstructedSource.get(position).put(childFieldName, formatVector(childFieldInfo, vectorValues));
            offsetPositionsIndex = position + 1;
        }
        sourceAsMap.put(parentFieldName, reconstructedSource);
    }

    private void injectObject(int docId, Map<String, Object> sourceAsMap) throws IOException {
        KNNVectorValues<?> vectorValues = KNNVectorValuesFactory.getVectorValues(
            childFieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
        if (vectorValues.docId() != docId && vectorValues.advance(docId) != docId) {
            return;
        }
        String[] fields = ParentChildHelper.splitPath(childFieldInfo.name);
        Map<String, Object> currentMap = sourceAsMap;
        for (int i = 0; i < fields.length - 1; i++) {
            String field = fields[i];
            currentMap = (Map<String, Object>) currentMap.computeIfAbsent(field, k -> new HashMap<>());
        }
        currentMap.put(fields[fields.length - 1], formatVector(childFieldInfo, vectorValues));
    }

    /**
     * Given a list of maps, map each map to a position in the nested list. This is used to help figure out where to put
     * the vectors back in the source.
     *
     * @param originals list of maps
     * @param firstChild first child docId
     * @param parent    parent docId
     * @return list of positions in the nested list
     * @throws IOException if there is an issue reading from the formats
     */
    private List<Integer> mapObjectsToPositionInNestedList(List<Map<String, Object>> originals, int firstChild, int parent)
        throws IOException {
        List<Integer> positions = new ArrayList<>();
        int offset = firstChild;
        for (Map<String, Object> docWithFields : originals) {
            int fieldMapping = mapToDocId(docWithFields, offset, parent);
            assert fieldMapping != -1;
            positions.add(fieldMapping);
            offset = fieldMapping + 1;
        }
        return positions;
    }

    /**
     * Given a doc as a map and the offset it has to be, find the ordinal of the first field that is greater than the
     * offset.
     *
     * @param doc    doc to find the ordinal for
     * @param offset offset to start searching from
     * @return id of the first field that is greater than the offset
     * @throws IOException if there is an issue reading from the formats
     */
    private int mapToDocId(Map<String, Object> doc, int offset, int parent) throws IOException {
        // For all the fields, we look for the first doc that matches any of the fields.
        int position = NO_MORE_DOCS;
        for (String key : doc.keySet()) {
            position = getLowestDocIdForField(ParentChildHelper.constructSiblingField(childFieldInfo.name, key), offset);
            if (position < parent) {
                break;
            }
        }

        // Advancing past the parent means something went wrong
        assert position < parent;
        return position;
    }

    /**
     * Get the lowest docId for a field that is greater than the offset.
     *
     * @param fieldToMatch field to find the lowest docId for
     * @param offset       offset to start searching from
     * @return lowest docId for the field that is greater than the offset. Returns {@link DocIdSetIterator#NO_MORE_DOCS} if doc cannot be found
     * @throws IOException if there is an issue reading from the formats
     */
    private int getLowestDocIdForField(String fieldToMatch, int offset) throws IOException {
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
                    iterator = derivedSourceReaders.getKnnVectorsReader().getFloatVectorValues(fieldInfo.name);
                    break;
                case BYTE:
                    iterator = derivedSourceReaders.getKnnVectorsReader().getByteVectorValues(fieldInfo.name);
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
        if (derivedSourceReaders.getFieldsProducer() == null) {
            return NO_MORE_DOCS;
        }
        Terms terms = derivedSourceReaders.getFieldsProducer().terms(FieldNamesFieldMapper.NAME);
        if (terms == null) {
            return NO_MORE_DOCS;
        }
        TermsEnum fieldNameFieldsTerms = terms.iterator();
        BytesRef fieldToMatchRef = new BytesRef(fieldInfo.name);
        PostingsEnum postingsEnum = null;
        while (fieldNameFieldsTerms.next() != null) {
            BytesRef currentTerm = fieldNameFieldsTerms.term();
            if (currentTerm.bytesEquals(fieldToMatchRef)) {
                postingsEnum = fieldNameFieldsTerms.postings(null);
                break;
            }
        }
        if (postingsEnum == null) {
            return NO_MORE_DOCS;
        }
        return postingsEnum.advance(offset);
    }
}
