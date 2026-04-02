/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.ingest.ConfigurationUtils;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.script.Script;
import org.opensearch.search.pipeline.AbstractProcessor;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchRequestProcessor;

import java.util.List;
import java.util.Map;

/**
 * Search request processor that implements the MUVERA two-phase retrieval pattern.
 *
 * The user sends a standard script_score query with lateInteractionScore and multi-vectors
 * in the script params. The processor intercepts it, extracts the multi-vectors, encodes
 * them via MUVERA into an FDE vector, and replaces the inner query (typically match_all)
 * with a knn query on the FDE field for fast ANN prefetch. The script_score wrapper and
 * its lateInteractionScore script remain untouched for the rerank phase.
 *
 * User sends:
 * <pre>
 * POST /my-index/_search?search_pipeline=muvera_pipeline
 * {
 *   "query": {
 *     "script_score": {
 *       "query": { "match_all": {} },
 *       "script": {
 *         "source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
 *         "params": {
 *           "query_vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
 *           "space_type": "innerproduct"
 *         }
 *       }
 *     }
 *   }
 * }
 * </pre>
 *
 * Configuration:
 * <pre>
 * {
 *   "muvera_query": {
 *     "target_field": "muvera_fde",        (required)
 *     "dim": 128,                          (required - input vector dimension)
 *     "k_sim": 4,                          (default: 4)
 *     "dim_proj": 8,                       (default: 8)
 *     "r_reps": 20,                        (default: 20)
 *     "seed": 42,                          (default: 42)
 *     "oversample_factor": 4               (default: 4)
 *     "fde_dimension": 2560                (optional - validates against computed value)
 *   }
 * }
 * </pre>
 */
public class MuveraSearchRequestProcessor extends AbstractProcessor implements SearchRequestProcessor {

    public static final String TYPE = "muvera_query";
    static final String QUERY_VECTORS_PARAM = "query_vectors";
    private static final Logger logger = LogManager.getLogger(MuveraSearchRequestProcessor.class);

    private final String targetField;
    private final MuveraEncoder encoder;
    private final int dim;
    private final int fdeDimension;
    private final int oversampleFactor;

    MuveraSearchRequestProcessor(
        String tag,
        String description,
        boolean ignoreFailure,
        String targetField,
        MuveraEncoder encoder,
        int dim,
        int fdeDimension,
        int oversampleFactor
    ) {
        super(tag, description, ignoreFailure);
        this.targetField = targetField;
        this.encoder = encoder;
        this.dim = dim;
        this.fdeDimension = fdeDimension;
        this.oversampleFactor = oversampleFactor;
    }

    /**
     * Extracts the Script from a ScriptScoreQueryBuilder by serializing to XContent and parsing
     * the script section. This avoids reflection on the private 'script' field.
     */
    static Script extractScript(ScriptScoreQueryBuilder scriptScoreQuery) throws Exception {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        scriptScoreQuery.toXContent(xContentBuilder, ToXContent.EMPTY_PARAMS);
        try (
            XContentParser parser = XContentType.JSON.xContent()
                .createParser(
                    NamedXContentRegistry.EMPTY,
                    LoggingDeprecationHandler.INSTANCE,
                    BytesReference.bytes(xContentBuilder).streamInput()
                )
        ) {
            // Structure: { "script_score": { "query": {...}, "script": {...}, ... } }
            parser.nextToken(); // START_OBJECT (outer)
            while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                if ("script_score".equals(parser.currentName())) {
                    parser.nextToken(); // START_OBJECT for script_score body
                    while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                        if ("script".equals(parser.currentName())) {
                            parser.nextToken(); // START_OBJECT for script body
                            return Script.parse(parser);
                        } else {
                            parser.skipChildren();
                        }
                    }
                } else {
                    parser.skipChildren();
                }
            }
        }
        throw new IllegalStateException("Failed to extract script from ScriptScoreQueryBuilder via XContent serialization");
    }

    @Override
    public SearchRequest processRequest(SearchRequest request) throws Exception {
        ScriptScoreQueryBuilder scriptScoreQuery = validateUserRequest(request);
        if (scriptScoreQuery == null) {
            return request;
        }

        Script script = extractScript(scriptScoreQuery);
        double[][] multiVectors = extractRequestParams(script);
        if (multiVectors == null) {
            return request;
        }

        return createKnnRequest(request, scriptScoreQuery, script, multiVectors);
    }

    /**
     * Validates the search request contains a script_score query.
     * @return the ScriptScoreQueryBuilder if valid, null if the request should be passed through unchanged
     */
    private ScriptScoreQueryBuilder validateUserRequest(SearchRequest request) {
        if (request.source() == null || request.source().query() == null) {
            return null;
        }
        QueryBuilder query = request.source().query();
        if (query instanceof ScriptScoreQueryBuilder == false) {
            return null;
        }
        return (ScriptScoreQueryBuilder) query;
    }

    /**
     * Extracts and validates multi-vector query parameters from the script.
     * @return the parsed multi-vectors as double[][], or null if query_vectors param is not present
     */
    private double[][] extractRequestParams(Script script) {
        Map<String, Object> params = script.getParams();
        if (params == null || params.containsKey(QUERY_VECTORS_PARAM) == false) {
            return null;
        }

        Object queryVectorsObj = params.get(QUERY_VECTORS_PARAM);
        if (queryVectorsObj instanceof List == false) {
            throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] in script params must be a list of vectors");
        }

        List<?> outerList = (List<?>) queryVectorsObj;
        int numTokens = outerList.size();
        if (numTokens == 0) {
            throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] must not be empty");
        }

        double[][] multiVectors = new double[numTokens][dim];
        for (int t = 0; t < numTokens; t++) {
            Object vecObj = outerList.get(t);
            if (vecObj instanceof List == false) {
                throw new IllegalArgumentException(
                    "[" + QUERY_VECTORS_PARAM + "] element at index [" + t + "] must be a list of numbers"
                );
            }
            List<?> tokenVec = (List<?>) vecObj;
            if (tokenVec.size() != dim) {
                throw new IllegalArgumentException(
                    "["
                        + QUERY_VECTORS_PARAM
                        + "] vector at index ["
                        + t
                        + "] has dimension ["
                        + tokenVec.size()
                        + "], expected ["
                        + dim
                        + "]. Check the 'dim' parameter in your MUVERA search processor configuration."
                );
            }
            for (int d = 0; d < dim; d++) {
                Object numObj = tokenVec.get(d);
                if (numObj instanceof Number == false) {
                    throw new IllegalArgumentException(
                        "[" + QUERY_VECTORS_PARAM + "] element at [" + t + "][" + d + "] is not a number"
                    );
                }
                multiVectors[t][d] = ((Number) numObj).doubleValue();
            }
        }
        return multiVectors;
    }

    /**
     * Encodes query multi-vectors into FDE and rewrites the search request with a KNN query.
     */
    private SearchRequest createKnnRequest(
        SearchRequest request,
        ScriptScoreQueryBuilder scriptScoreQuery,
        Script script,
        double[][] multiVectors
    ) {
        float[] queryFde = encoder.processQuery(multiVectors);

        if (queryFde.length != fdeDimension) {
            throw new IllegalStateException(
                "MUVERA encoder produced query FDE of dimension ["
                    + queryFde.length
                    + "] but expected ["
                    + fdeDimension
                    + "]. This should not happen — please report this as a bug."
            );
        }

        int resultSize = request.source().size() > 0 ? request.source().size() : 10;
        int prefetchK = Math.min(resultSize * oversampleFactor, 10_000);

        KNNQueryBuilder knnQuery = KNNQueryBuilder.builder().fieldName(targetField).vector(queryFde).k(prefetchK).build();

        ScriptScoreQueryBuilder rewritten = new ScriptScoreQueryBuilder(knnQuery, script);
        rewritten.boost(scriptScoreQuery.boost());
        if (scriptScoreQuery.queryName() != null) {
            rewritten.queryName(scriptScoreQuery.queryName());
        }
        if (scriptScoreQuery.getMinScore() != null) {
            rewritten.setMinScore(scriptScoreQuery.getMinScore());
        }

        request.source().query(rewritten);
        return request;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    public static class Factory implements Processor.Factory<SearchRequestProcessor> {

        @Override
        public MuveraSearchRequestProcessor create(
            Map<String, Processor.Factory<SearchRequestProcessor>> processorFactories,
            String tag,
            String description,
            boolean ignoreFailure,
            Map<String, Object> config,
            PipelineContext pipelineContext
        ) throws Exception {
            String targetField = ConfigurationUtils.readStringProperty(TYPE, tag, config, "target_field");

            // dim is required — no sensible default since it depends on the embedding model
            Integer dimValue = ConfigurationUtils.readIntProperty(TYPE, tag, config, "dim", null);
            if (dimValue == null) {
                throw ConfigurationUtils.newConfigurationException(
                    TYPE,
                    tag,
                    "dim",
                    "required property is missing. Set this to your embedding model's vector dimension (e.g. 128)."
                );
            }
            int dim = dimValue;

            int kSim = ConfigurationUtils.readIntProperty(TYPE, tag, config, "k_sim", 4);
            int dimProj = ConfigurationUtils.readIntProperty(TYPE, tag, config, "dim_proj", 8);
            int rReps = ConfigurationUtils.readIntProperty(TYPE, tag, config, "r_reps", 20);
            long seed = MuveraProcessorUtils.readLongProperty(TYPE, tag, config, "seed", 42L);
            int oversampleFactor = ConfigurationUtils.readIntProperty(TYPE, tag, config, "oversample_factor", 4);
            if (oversampleFactor <= 0) {
                throw ConfigurationUtils.newConfigurationException(
                    TYPE,
                    tag,
                    "oversample_factor",
                    "must be positive, got: " + oversampleFactor
                );
            }

            MuveraEncoder encoder = new MuveraEncoder(dim, kSim, dimProj, rReps, seed);
            int computedDimension = encoder.getEmbeddingSize();

            // Validate fde_dimension if provided
            Integer userDimension = ConfigurationUtils.readIntProperty(TYPE, tag, config, "fde_dimension", null);
            if (userDimension != null && userDimension != computedDimension) {
                throw new IllegalArgumentException(
                    "["
                        + TYPE
                        + "] processor ["
                        + tag
                        + "] fde_dimension ["
                        + userDimension
                        + "] does not match computed dimension ["
                        + computedDimension
                        + "] (r_reps="
                        + rReps
                        + " * 2^k_sim="
                        + (1 << kSim)
                        + " * dim_proj="
                        + dimProj
                        + ")"
                );
            }

            // Log the FDE dimension so the user can verify it matches their index mapping
            logger.info(
                "[{}] processor [{}]: computed FDE dimension = {} (r_reps={} * 2^k_sim={} * dim_proj={}). "
                    + "This must match the 'dimension' in your knn_vector field mapping for [{}].",
                TYPE,
                tag,
                computedDimension,
                rReps,
                (1 << kSim),
                dimProj,
                targetField
            );

            // Auto-generate description if not provided
            if (description == null || description.isEmpty()) {
                description = "MUVERA query encoder: dim="
                    + dim
                    + ", fde_dimension="
                    + computedDimension
                    + " (r_reps="
                    + rReps
                    + ", k_sim="
                    + kSim
                    + ", dim_proj="
                    + dimProj
                    + ", seed="
                    + seed
                    + ", oversample_factor="
                    + oversampleFactor
                    + ")";
            }

            return new MuveraSearchRequestProcessor(
                tag,
                description,
                ignoreFailure,
                targetField,
                encoder,
                dim,
                computedDimension,
                oversampleFactor
            );
        }
        }
    }
}
