/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.DynamicTemplateTypeHandler;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.Map;

/**
 * k-NN implementation of {@link DynamicTemplateTypeHandler}.
 *
 * <p>Called by core when a dynamic template with {@code match_mapping_type: "knn_vector"} matches
 * an unmapped field. Runs <em>before</em> {@code TypeParser.parse()} so that any required parameters
 * are present when the mapper is constructed.
 *
 * <p>The only adjustment made is injecting {@code dimension} from the array length when the user
 * did not specify it in the template. Without dimension, {@code KNNVectorFieldMapper.TypeParser}
 * would throw "Dimension value missing."
 *
 * <p>If the user already specified {@code dimension} (or a {@code model_id} that supplies it) in the
 * template mapping config, the existing value is preserved — this handler never overwrites a
 * user-provided dimension, and it never even opens a parser in that case. Because a complete config
 * opens no parser, core can also validate such templates eagerly at index-creation time.
 *
 * <p>If the field value is not an array (e.g. the template matched via a path pattern on a
 * non-array field), the handler does nothing — the TypeParser will validate the config and reject
 * it if dimension is truly required.
 */
public class KNNDynamicTemplateTypeHandler implements DynamicTemplateTypeHandler {

    /**
     * Injects {@code dimension} into the mapping config from the array length if not already present.
     * Only creates a parser when dimension is missing — fully-specified templates open nothing.
     *
     * @param mappingConfig the mutable mapping config from the matched template
     * @param fieldValueParser produces a fresh parser positioned at the field value's first token
     */
    @Override
    public void adjustMappingConfig(Map<String, Object> mappingConfig, FieldValueParserSupplier fieldValueParser) throws IOException {
        // The type is implied by match_mapping_type: "knn_vector", so a template may omit it from the
        // mapping block (or omit the block entirely). Inject it here so the TypeParser always receives a
        // complete config — the plugin owns its own type, core stays type-agnostic.
        mappingConfig.putIfAbsent("type", KNNVectorFieldMapper.CONTENT_TYPE);
        // A complete config needs no data-derived parameter, so we must not open the parser: doing so
        // would defer index-creation-time validation, and injecting a data-derived dimension alongside
        // a model_id is rejected by the TypeParser.
        if (isConfigComplete(mappingConfig)) {
            return;
        }
        try (XContentParser parser = fieldValueParser.get()) {
            if (parser.currentToken() != XContentParser.Token.START_ARRAY) {
                return;
            }
            int count = 0;
            while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                count++;
            }
            mappingConfig.put(KNNConstants.DIMENSION, count);
        }
    }

    /**
     * A knn_vector template is fully specified when the dimension is given directly, or when a
     * {@code model_id} supplies it. In both cases the mapper can be built without inspecting a
     * document, so core can validate the template eagerly at index-creation time.
     */
    @Override
    public boolean isConfigComplete(Map<String, Object> mappingConfig) {
        return mappingConfig.containsKey(KNNConstants.DIMENSION) || mappingConfig.containsKey(KNNConstants.MODEL_ID);
    }
}
