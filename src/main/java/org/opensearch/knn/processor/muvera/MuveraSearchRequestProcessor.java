/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.ingest.ConfigurationUtils;
import org.opensearch.search.pipeline.AbstractProcessor;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchRequestProcessor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Search request processor that implements the MUVERA query encoding for template-based retrieval.
 *
 * The user sends a template query containing a script_score with a KNN placeholder and
 * lateInteractionScore for reranking. The processor extracts the multi-vectors from the
 * script params in the ext section, encodes them via MUVERA into an FDE vector, and sets
 * the FDE as a pipeline context attribute. The template query resolves the ${target_field}
 * placeholder with the FDE vector during query rewrite.
 *
 * User sends:
 * <pre>
 * POST /my-index/_search?search_pipeline=muvera_pipeline
 * {
 *   "query": {
 *     "template": {
 *       "script_score": {
 *         "query": {
 *           "knn": {
 *             "muvera_fde": {
 *               "vector": "${muvera_fde}",
 *               "k": 40
 *             }
 *           }
 *         },
 *         "script": {
 *           "source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
 *           "params": {
 *             "query_vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
 *             "space_type": "innerproduct"
 *           }
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
 *     "target_field": "muvera_fde",        (required - also used as the template variable name)
 *     "dim": 128,                          (required - input vector dimension)
 *     "k_sim": 4,                          (default: 4)
 *     "dim_proj": 8,                       (default: 8)
 *     "r_reps": 20,                        (default: 20)
 *     "seed": 42,                          (default: 42)
 *     "query_vectors_field": "ext.muvera.query_vectors"  (default - JSON path to multi-vectors in request)
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
    private final String queryVectorsField;

    MuveraSearchRequestProcessor(
        String tag,
        String description,
        boolean ignoreFailure,
        String targetField,
        MuveraEncoder encoder,
        int dim,
        int fdeDimension,
        String queryVectorsField
    ) {
        super(tag, description, ignoreFailure);
        this.targetField = targetField;
        this.encoder = encoder;
        this.dim = dim;
        this.fdeDimension = fdeDimension;
        this.queryVectorsField = queryVectorsField;
    }

    @Override
    public SearchRequest processRequest(SearchRequest request, PipelineProcessingContext requestContext) throws Exception {
        if (request.source() == null) {
            return request;
        }

        // Extract multi-vectors from the template query content
        double[][] multiVectors = extractQueryVectors(request);
        if (multiVectors == null) {
            return request;
        }

        // Encode query multi-vectors into FDE
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

        // Convert float[] to List<Float> for JSON serialization in template resolution
        List<Float> fdeList = new ArrayList<>(queryFde.length);
        for (float v : queryFde) {
            fdeList.add(v);
        }

        // Set the FDE vector as a pipeline context attribute.
        // The template query resolves ${target_field} from these attributes during query rewrite.
        requestContext.setAttribute(targetField, fdeList);

        return request;
    }

    @Override
    public SearchRequest processRequest(SearchRequest request) throws Exception {
        // This method is called when no PipelineProcessingContext is available.
        // Template query resolution requires context, so we can't resolve placeholders here.
        return request;
    }

    /**
     * Extracts multi-vector query parameters from the request's ext section.
     * The field path is configured via query_vectors_field (default: "ext.muvera.query_vectors").
     */
    @SuppressWarnings("unchecked")
    private double[][] extractQueryVectors(SearchRequest request) {
        if (request.source().query() == null) {
            return null;
        }

        QueryBuilder query = request.source().query();

        // Check if it's a template query and extract content directly
        if (query instanceof org.opensearch.index.query.TemplateQueryBuilder) {
            org.opensearch.index.query.TemplateQueryBuilder templateQuery =
                (org.opensearch.index.query.TemplateQueryBuilder) query;
            Map<String, Object> content = templateQuery.getContent();
            Object queryVectorsObj = extractQueryVectorsFromContent(content);
            if (queryVectorsObj != null) {
                return parseMultiVectors(queryVectorsObj);
            }
        }

        return null;
    }

    /**
     * Navigates the template content map to find query_vectors in script_score params.
     * Path: script_score -> script -> params -> query_vectors
     */
    @SuppressWarnings("unchecked")
    private Object extractQueryVectorsFromContent(Map<String, Object> content) {
        if (content == null) return null;

        Map<String, Object> scriptScore = (Map<String, Object>) content.get("script_score");
        if (scriptScore == null) return null;

        Map<String, Object> script = (Map<String, Object>) scriptScore.get("script");
        if (script == null) return null;

        Map<String, Object> params = (Map<String, Object>) script.get("params");
        if (params == null) return null;

        return params.get(QUERY_VECTORS_PARAM);
    }

    /**
     * Parses and validates multi-vector input from a raw object (List of List of Number).
     */
    private double[][] parseMultiVectors(Object queryVectorsObj) {
        if (queryVectorsObj instanceof List == false) {
            throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] must be a list of vectors");
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
            String queryVectorsField = ConfigurationUtils.readStringProperty(
                TYPE, tag, config, "query_vectors_field", "ext.muvera.query_vectors"
            );

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
                queryVectorsField
            );
        }
    }
}
