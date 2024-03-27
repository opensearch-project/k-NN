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
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.search.SearchHit;
import org.opensearch.search.fetch.FetchContext;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.FetchSubPhaseProcessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;


/**
 * Fetch sub phase which pull data from doc values.
 * and fulfill the value into source map
 */
public class KNNFetchSubPhase implements FetchSubPhase {
    private static Logger logger = LogManager.getLogger(KNNFetchSubPhase.class);

    @Override
    public FetchSubPhaseProcessor getProcessor(FetchContext fetchContext) throws IOException {
        if (!KNNSettings.isKNNSyntheticEnabled(fetchContext.getIndexName())) {
            return null;
        }
        MapperService mapperService = fetchContext.mapperService();

        List<DocValueField> fields = new ArrayList<>();
        for (MappedFieldType mappedFieldType : mapperService.fieldTypes()) {
            if (mappedFieldType != null && mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType) {
                String fieldName = mappedFieldType.name();
                ValueFetcher fetcher = new DocValueFetcher(mappedFieldType.docValueFormat(null, null),
                        fetchContext.searchLookup().doc().getForField(mappedFieldType));
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
                //when source is disabled, return
                return;
            }

            if (hasNested) {
                //TODO handle nested field
                logger.debug("Use nested:" + hasNested);
            }
            for (DocValueField f : fields) {
                if (maps.containsKey(f.field)) {
                    continue;
                }
                maps.put(f.field, f.fetcher.fetchValues(hitContext.sourceLookup()));
            }
            BytesStreamOutput streamOutput = new BytesStreamOutput(BYTES_PER_KILOBYTES);
            XContentBuilder builder = new XContentBuilder(XContentType.JSON.xContent(), streamOutput);
            builder.value(maps);
            hitContext.hit().sourceRef(BytesReference.bytes(builder));
        }
    }

    private static class DocValueField {
        private final String field;
        private final ValueFetcher fetcher;

        DocValueField(String field, ValueFetcher fetcher) {
            this.field = field;
            this.fetcher = fetcher;
        }
    }
}
