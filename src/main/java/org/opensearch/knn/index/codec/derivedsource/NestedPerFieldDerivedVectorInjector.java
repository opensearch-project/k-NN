/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
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
public class NestedPerFieldDerivedVectorInjector implements PerFieldDerivedVectorInjector {

    private final FieldInfo childFieldInfo;
    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;

    @Override
    public void inject(int parentDocId, Map<String, Object> sourceAsMap) throws IOException {
        // Setup the iterator. Return if not-relevant
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

        // Initializes the parent field so that there is a map to put each of the children
        Object originalParentValue = sourceAsMap.get(parentFieldName);
        List<Map<String, Object>> reconstructedSource;
        if (originalParentValue instanceof Map) {
            reconstructedSource = new ArrayList<>(List.of((Map<String, Object>) originalParentValue));
        } else {
            reconstructedSource = (List<Map<String, Object>>) originalParentValue;
        }

        // Contains the positions of existing objects in the map. This is used to help figure out the best play to put back the vectors
        List<Integer> positions = mapObjectsToPositionInSource(
            reconstructedSource,
            nestedPerFieldParentToDocIdIterator.firstChild(),
            parentDocId
        );

        // Finally, inject children for the document into the source. This code is non-trivial because filtering out
        // the vectors during write could mean that children docs disappear from the source. So, to properly put
        // everything back, we need to igure out where the existing fields in the original map to
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
            int docId = vectorValues.docId();
            if (docId >= parentDocId) {
                break;
            }
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
            reconstructedSource.get(position).put(childFieldName, vectorValues.conditionalCloneVector());
            offsetPositionsIndex = position + 1;
        }
        sourceAsMap.put(parentFieldName, reconstructedSource);
    }

    private List<Integer> mapObjectsToPositionInSource(List<Map<String, Object>> originals, int firstChild, int parent) throws IOException {
        List<Integer> positions = new ArrayList<>();
        int offset = firstChild;
        for (Map<String, Object> docWithFields : originals) {
            int fieldMapping = docToOrdinal(docWithFields, offset, parent);
            assert fieldMapping != -1;
            positions.add(fieldMapping);
            offset = fieldMapping + 1;
        }
        return positions;
    }

    // Offset is first eligible object
    private Integer docToOrdinal(Map<String, Object> doc, int offset, int parent) throws IOException {
        String keyToCheck = doc.keySet().iterator().next();
        int position = getFieldsForDoc(keyToCheck, offset);
        // Advancing past the parent means something went horribly wrong
        assert position < parent;
        return position;
    }

    private int getFieldsForDoc(String fieldToMatch, int offset) throws IOException {
        // TODO: Fix this up to follow
        // https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/FieldExistsQuery.java#L170-L218.
        // In a perfect world, it would try everything and fall through to the field exists stuff
        FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(
            ParentChildHelper.constructSiblingField(childFieldInfo.name, fieldToMatch)
        );
        DocIdSetIterator iterator = null;
        if (fieldInfo != null) {
            switch (fieldInfo.getDocValuesType()) {
                case NONE:
                    break;
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
                default:
                    throw new AssertionError();
            }
        }
        if (iterator != null) {
            return iterator.advance(offset);
        }

        Terms terms = derivedSourceReaders.getFieldsProducer().terms(FieldNamesFieldMapper.NAME);
        TermsEnum fieldNameFieldsTerms = terms.iterator();
        BytesRef fieldToMatchRef = new BytesRef(fieldToMatch);
        PostingsEnum postingsEnum = null;
        while (fieldNameFieldsTerms.next() != null) {
            BytesRef currentTerm = fieldNameFieldsTerms.term();
            if (currentTerm.bytesEquals(fieldToMatchRef)) {
                postingsEnum = fieldNameFieldsTerms.postings(null);
                break;
            }
        }
        assert postingsEnum != null;
        return postingsEnum.advance(offset);
    }
}
