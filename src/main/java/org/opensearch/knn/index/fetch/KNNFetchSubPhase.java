/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.fetch;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.lucene.search.Queries;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.search.SearchHit;
import org.opensearch.search.fetch.FetchContext;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.FetchSubPhaseProcessor;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.lookup.SourceLookup;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;

/**
 * Fetch sub phase which pull data from doc values.
 * and fulfill the value into source map
 */
@Log4j2
public class KNNFetchSubPhase implements FetchSubPhase {

    @Override
    public FetchSubPhaseProcessor getProcessor(FetchContext fetchContext) throws IOException {
        IndexSettings indexSettings = fetchContext.getIndexSettings();
        if (!KNNSettings.isKNNSyntheticSourceEnabled(indexSettings)) {
            log.debug("Synthetic is disabled for index: {}", fetchContext.getIndexName());
            return null;
        }
        MapperService mapperService = fetchContext.mapperService();

        List<DocValueField> fields = new ArrayList<>();
        for (MappedFieldType mappedFieldType : mapperService.fieldTypes()) {
            if (mappedFieldType != null && mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType) {
                String fieldName = mappedFieldType.name();
                ValueFetcher fetcher = new DocValueFetcher(
                    mappedFieldType.docValueFormat(null, null),
                    fetchContext.searchLookup().doc().getForField(mappedFieldType)
                );
                fields.add(new DocValueField(fieldName, fetcher));
            }
        }
        return new KNNFetchSubPhaseProcessor(fetchContext, fields);
    }

    @AllArgsConstructor
    @Getter
    class KNNFetchSubPhaseProcessor implements FetchSubPhaseProcessor {

        private final FetchContext fetchContext;
        private final List<DocValueField> fields;

        @Override
        public void setNextReader(LeafReaderContext readerContext) throws IOException {
            for (DocValueField f : fields) {
                f.fetcher.setNextReader(readerContext);
            }
        }

        @Override
        public void process(HitContext hitContext) throws IOException {
            MapperService mapperService = fetchContext.mapperService();
            final boolean hasNested = mapperService.hasNested();
            SearchHit hit = hitContext.hit();
            Map<String, Object> maps = hit.getSourceAsMap();
            if (maps == null) {
                // when source is disabled, return
                return;
            }

            if (hasNested) {
                syntheticNestedDocValues(mapperService, hitContext, maps);
            }
            for (DocValueField f : fields) {
                if (maps.containsKey(f.field)) {
                    continue;
                }
                List<Object> docValuesSource = f.fetcher.fetchValues(hitContext.sourceLookup());
                if (docValuesSource.size() > 0) {
                    maps.put(f.field, docValuesSource.get(0));
                }
            }
            BytesStreamOutput streamOutput = new BytesStreamOutput(BYTES_PER_KILOBYTES);
            XContentBuilder builder = new XContentBuilder(XContentType.JSON.xContent(), streamOutput);
            builder.value(maps);
            hitContext.hit().sourceRef(BytesReference.bytes(builder));
        }

        protected void syntheticNestedDocValues(MapperService mapperService, HitContext hitContext, Map<String, Object> maps)
            throws IOException {
            DocumentMapper documentMapper = mapperService.documentMapper();
            Map<String, ObjectMapper> mapperMap = documentMapper.objectMappers();
            SearchHit hit = hitContext.hit();

            for (ObjectMapper objectMapper : mapperMap.values()) {
                if (objectMapper == null) {
                    continue;
                }
                if (!objectMapper.nested().isNested()) {
                    continue;
                }
                String path = objectMapper.fullPath();
                for (DocValueField f : fields) {
                    if (!f.field.startsWith(path)) {
                        continue;
                    }
                    if (!maps.containsKey(path)) {
                        continue;
                    }

                    // path to nested field:
                    Object nestedObj = maps.get(path);
                    if (!(nestedObj instanceof ArrayList)) {
                        continue;
                    }
                    // nested array in one nested path
                    ArrayList nestedDocList = (ArrayList) nestedObj;

                    log.debug(
                        "object mapper: nested:"
                            + objectMapper.nested().isNested()
                            + " Value:"
                            + objectMapper.fullPath()
                            + " field:"
                            + f.field
                    );

                    Query parentFilter = Queries.newNonNestedFilter();
                    Query childFilter = objectMapper.nestedTypeFilter();
                    ContextIndexSearcher searcher = fetchContext.searcher();
                    final Weight childWeight = searcher.createWeight(searcher.rewrite(childFilter), ScoreMode.COMPLETE_NO_SCORES, 1f);

                    LeafReaderContext subReaderContext = hitContext.readerContext();
                    Scorer childScorer = childWeight.scorer(subReaderContext);
                    DocIdSetIterator childIter = childScorer.iterator();
                    BitSet parentBits = fetchContext.getQueryShardContext().bitsetFilter(parentFilter).getBitSet(subReaderContext);

                    int currentParent = hit.docId() - subReaderContext.docBase;
                    int previousParent = parentBits.prevSetBit(currentParent - 1);
                    int childDocId = childIter.advance(previousParent + 1);
                    SourceLookup nestedVecSourceLookup = new SourceLookup();

                    // when nested field only have vector field and exclude source, list is empty
                    boolean isEmpty = nestedDocList.isEmpty();

                    for (int offset = 0; childDocId < currentParent && childDocId != DocIdSetIterator.NO_MORE_DOCS; childDocId = childIter
                        .nextDoc(), offset++) {
                        nestedVecSourceLookup.setSegmentAndDocument(subReaderContext, childDocId);
                        List<Object> nestedVecDocValuesSource = f.fetcher.fetchValues(nestedVecSourceLookup);
                        if (nestedVecDocValuesSource == null || nestedVecDocValuesSource.isEmpty()) {
                            continue;
                        }
                        if (isEmpty) {
                            nestedDocList.add(new HashMap<String, Object>());
                        }
                        if (offset < nestedDocList.size()) {
                            Object o2 = nestedDocList.get(offset);
                            if (o2 instanceof Map) {
                                Map<String, Object> o2map = (Map<String, Object>) o2;
                                String suffix = f.field.substring(path.length() + 1);
                                o2map.put(suffix, nestedVecDocValuesSource.get(0));
                            }
                        } else {
                            /**
                             * TODO nested field partial doc only have vector and source exclude
                             * this source map nestedDocList would out-of-order, can not fill the vector into right offset
                             * "nested_field" : [
                             *    {"nested_vector": [2.6, 2.6]},
                             *    {"nested_numeric": 2, "nested_vector": [3.1, 2.3]}
                             *  ]
                             */

                            throw new UnsupportedOperationException(
                                String.format(
                                    "\"Nested Path \"%s\" in Field \"%s\" with _ID \"%s\" can not be empty\"",
                                    path,
                                    f.field,
                                    hit.getId()
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    @Getter
    public static class DocValueField {
        private final String field;
        private final ValueFetcher fetcher;

        DocValueField(String field, ValueFetcher fetcher) {
            this.field = field;
            this.fetcher = fetcher;
        }
    }
}
