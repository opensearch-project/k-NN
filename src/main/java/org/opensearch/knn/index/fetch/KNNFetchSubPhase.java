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
import org.opensearch.common.document.DocumentField;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.search.SearchHit;
import org.opensearch.search.fetch.FetchContext;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.FetchSubPhaseProcessor;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;


public class KNNFetchSubPhase implements FetchSubPhase {
    private static Logger logger = LogManager.getLogger(KNNFetchSubPhase.class);


    @Override
    public FetchSubPhaseProcessor getProcessor(FetchContext fetchContext) throws IOException {
        return null;
    }

    @AllArgsConstructor
    @Getter
    class KNNFetchSubPhaseProcessor implements FetchSubPhaseProcessor {

        private final FetchContext fetchContext;


        @Override
        public void setNextReader(LeafReaderContext leafReaderContext) throws IOException {

        }

        @Override
        public void process(HitContext hitContext) throws IOException {
            SearchHit hit = hitContext.hit();
            Map<String, DocumentField> fields = hit.getFields();
            MapperService mapperService = fetchContext.mapperService();
            Map<String, Object> maps = hit.getSourceAsMap();

            for (Map.Entry<String, DocumentField> fieldsEntry : fields.entrySet()) {
                String fieldName = fieldsEntry.getKey();
                MappedFieldType mappedFieldType = mapperService.fieldType(fieldName);
                if (mappedFieldType != null && mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType) {
                    maps.put(fieldName, fieldsEntry.getValue());
                }
            }

            //TODO process nested
            BytesStreamOutput streamOutput = new BytesStreamOutput(BYTES_PER_KILOBYTES);
            XContentBuilder builder = new XContentBuilder(XContentType.JSON.xContent(), streamOutput);
            builder.value(maps);
            hitContext.hit().sourceRef(BytesReference.bytes(builder));
        }
    }
}
