/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Injector class for nested fields and object fields. The class assumes that there will only be one level of nesting
 * and that any vector path will have only one parent (i.e. parent.vector is supported but grandparent.parent.vector is
 * not)
 */
@Log4j2
public class NestedPerFieldDerivedVectorInjector extends AbstractPerFieldDerivedVectorInjector {

    private final FieldInfo childFieldInfo;
    private final DerivedSourceReaders derivedSourceReaders;
    private final DerivedSourceLuceneHelper derivedSourceLuceneHelper;
    private final SegmentReadState segmentReadState;

    /**
     *
     * @param childFieldInfo FieldInfo of the child field
     * @param derivedSourceReaders Readers for access segment info
     * @param segmentReadState Segment read stats
     */
    public NestedPerFieldDerivedVectorInjector(
        FieldInfo childFieldInfo,
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState
    ) {
        this.childFieldInfo = childFieldInfo;
        this.derivedSourceReaders = derivedSourceReaders;
        this.segmentReadState = segmentReadState;
        this.derivedSourceLuceneHelper = new DerivedSourceLuceneHelper(derivedSourceReaders, segmentReadState);
    }

    @Override
    public void inject(int docId, Map<String, Object> sourceAsMap) throws IOException {
        KNNVectorValues<?> vectorValues = KNNVectorValuesFactory.getVectorValues(
            childFieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );

        // TODO Can remove
        // If the doc has the field, then it is just an object field.
        if (derivedSourceLuceneHelper.fieldExists(childFieldInfo.name, docId)) {
            injectObject(docId, sourceAsMap, vectorValues);
            return;
        }

        // If the doc doesnt have the field, either the field is a nested field or it is an object field that is
        // just not present for the doc. Regardless, we will treat as nested and do nothing if it is just missing the
        // field.
        injectNested(docId, sourceAsMap, vectorValues);
    }

    private void injectObject(int docId, Map<String, Object> sourceAsMap, KNNVectorValues<?> vectorValues) throws IOException {
        // Check if a vector is actually present for this value
        if (vectorValues.docId() != docId && vectorValues.advance(docId) != docId) {
            return;
        }
        DerivedSourceMapHelper.injectVector(
            sourceAsMap,
            formatVector(childFieldInfo, vectorValues::getVector, vectorValues::conditionalCloneVector),
            childFieldInfo.name
        );
    }

    private void injectNested(int parentDocId, Map<String, Object> sourceAsMap, KNNVectorValues<?> vectorValues) throws IOException {
        // The first child represents the first child document of the parent. This does not mean that this child is
        // a document belonging to the current field that is being injected. Instead, it just means that the
        // parent doc has nested fields
        int firstChild = derivedSourceLuceneHelper.getFirstChild(parentDocId);
        if (firstChild == NO_MORE_DOCS) {
            return;
        }

        // We need to check if the parent field is a nested field.
        String childFieldName = ParentChildHelper.getChildField(childFieldInfo.name);
        String parentFieldName = ParentChildHelper.getParentField(childFieldInfo.name);
        if (derivedSourceLuceneHelper.isNestedParent(firstChild, parentDocId, parentFieldName) == false) {
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

        // TODO: Update this to call out that empty objects are removed from arrays.
        // In order to inject vectors into source for nested documents, we need to be able to map the existing
        // maps to document positions. This lets us know what place to put the vector back into. For example:
        // Assume we have the following document from the user and we are deriving the value for nested.vector
        // {
        // "nested": [
        // {
        // "text": "text1"
        // },
        // {
        // "vector": [vec1]
        // },
        // {
        // "vector": [vec2],
        // "text": "text2"
        // }
        // ]
        // }
        //
        // This would get filtered and serialized as:
        // {
        // "nested": [
        // {
        // "text": "text1"
        // },
        // {
        // "text": "text2"
        // }
        // ]
        // }
        //
        // We need to ensure that when we want to inject vec1 back, we create a new map and put it in between
        // the existing fields.
        //
        // To do this, we need to know what docs the existing 2 maps map to.
        List<Integer> docIdsInNestedList = mapObjectsInNestedListToDocIds(reconstructedSource, firstChild, parentDocId);

        // Finally, inject children for the document into the source.
        NestedPerFieldParentToChildDocIdIterator nestedPerFieldParentToChildDocIdIterator = new NestedPerFieldParentToChildDocIdIterator(
            parentDocId,
            firstChild,
            vectorValues
        );
        int offsetPositionsIndex = 0;
        while (nestedPerFieldParentToChildDocIdIterator.nextDoc() != NO_MORE_DOCS) {
            int docId = nestedPerFieldParentToChildDocIdIterator.docId();
            boolean isInsert = true;

            // Find the position in the nested source list of maps to put it back
            int position = docIdsInNestedList.size();
            for (int i = offsetPositionsIndex; i < docIdsInNestedList.size(); i++) {
                if (docId < docIdsInNestedList.get(i)) {
                    position = i;
                    break;
                }
                if (docId == docIdsInNestedList.get(i)) {
                    isInsert = false;
                    position = i;
                    break;
                }
            }

            // If we need to insert a new map, we do so here
            if (isInsert) {
                reconstructedSource.add(position, new HashMap<>());
                docIdsInNestedList.add(position, docId);
            }
            DerivedSourceMapHelper.injectVector(
                reconstructedSource.get(position),
                formatVector(
                    childFieldInfo,
                    nestedPerFieldParentToChildDocIdIterator::getVector,
                    nestedPerFieldParentToChildDocIdIterator::getVectorClone
                ),
                childFieldName
            );
            offsetPositionsIndex = position + 1;
        }
        sourceAsMap.put(parentFieldName, reconstructedSource);
    }

    /**
     * Given a list of maps, map each map to a doc id. This is used to help figure out where to put
     * the vectors back in the source. The assumption is that earlier objects in the list will have lower doc ids than
     * later objects in the map
     *
     * @param originals list of maps
     * @param offset Position to move iterators to identify the positions in the map
     * @param parent    parent docId
     * @return list of positions in the nested list
     * @throws IOException if there is an issue reading from the formats
     */
    private List<Integer> mapObjectsInNestedListToDocIds(List<Map<String, Object>> originals, int offset, int parent) throws IOException {
        // Starting at the offset, we iterate over all of maps in the list of maps and figure out what doc id they map
        // to.
        List<Integer> positions = new ArrayList<>();
        int currentOffset = offset;
        for (Map<String, Object> docWithFields : originals) {
            int fieldMapping = getDocIdOfMap(docWithFields, currentOffset, parent);
            positions.add(fieldMapping);
            currentOffset = fieldMapping + 1;
        }
        return positions;
    }

    /**
     * Given a doc as a map and the offset it has to be after, return the doc id that it must be
     *
     * @param doc    doc to find the docId for
     * @param offset offset to start searching from
     * @param parent parent docId
     * @return doc id the map must map to
     * @throws IOException if there is an issue reading from the formats
     */
    private int getDocIdOfMap(Map<String, Object> doc, int offset, int parent) throws IOException {
        // First, we identify a field that the doc in question must have
        FieldInfo fieldInfoOfDoc = getAnyMatchingFieldInfoForDoc(doc);
        assert fieldInfoOfDoc != null;

        // Get the first document on/after the offset that has this field.
        int firstMatchingDocWithField = derivedSourceLuceneHelper.getFirstDocWhereFieldExists(fieldInfoOfDoc.name, offset);
        // Advancing past the parent means something went wrong
        assert firstMatchingDocWithField < parent;

        // The field in question may be a nested field. In this case, we need to find the next parent on the same level
        // as the child to figure out where to put back the vector.
        int position = derivedSourceLuceneHelper.getNextDocIdWithParent(
            firstMatchingDocWithField,
            parent - 1,
            ParentChildHelper.getParentField(childFieldInfo.name)
        );
        // We should not advance past the parent
        assert position < parent;
        return position;
    }

    /**
     * For a given map, return a {@link FieldInfo} that the doc must have.
     *
     * @param doc source of the document
     * @return {@link FieldInfo} of any field the document must have; null if none are found
     */
    private FieldInfo getAnyMatchingFieldInfoForDoc(Map<String, Object> doc) {
        for (FieldInfo fieldInfo : segmentReadState.fieldInfos) {
            String extractedFieldName = ParentChildHelper.getChildField(fieldInfo.name);
            String parentFieldName = ParentChildHelper.getParentField(fieldInfo.name);
            if (extractedFieldName == null || !Objects.equals(parentFieldName, ParentChildHelper.getParentField(childFieldInfo.name))) {
                continue;
            }

            if (DerivedSourceMapHelper.fieldExists(doc, extractedFieldName)) {
                return fieldInfo;
            }
        }
        return null;
    }
}
