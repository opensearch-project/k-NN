/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.TemplateQueryBuilder;
import org.opensearch.ingest.ConfigurationUtils;
import org.opensearch.search.pipeline.AbstractProcessor;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchRequestProcessor;

import java.util.List;
import java.util.Map;

/**
 * Search request processor that implements the MUVERA query encoding for template-based retrieval.
 *
 * <p>The user sends a template query containing a script_score with a KNN placeholder and
 * a {@code lateInteractionScore} reranking script. The processor extracts the multi-vector
 * query from the script params, encodes it via MUVERA into a single FDE vector, and stores
 * the FDE as a {@link PipelineProcessingContext} attribute. The template query then resolves
 * the {@code ${target_field}} placeholder during query rewrite using that attribute, so the
 * KNN search runs with the encoded FDE and the script_score reranks the prefetched
 * candidates with exact MaxSim.
 *
 * <p>User sends:
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
 * <p>Configuration:
 * <pre>
 * {
 *   "muvera_query": {
 *     "target_field": "muvera_fde",        (required - also used as the template variable name)
 *     "dim": 128,                          (required - input vector dimension)
 *     "k_sim": 4,                          (default: 4)
 *     "dim_proj": 8,                       (default: 8)
 *     "r_reps": 20,                        (default: 20)
 *     "seed": 42,                          (default: 42)
 *     "fde_dimension": 2560                (optional - validates against computed value)
 *   }
 * }
 * </pre>
 */
@Log4j2
public class MuveraSearchRequestProcessor extends AbstractProcessor implements SearchRequestProcessor {

    /** Processor type identifier registered with the search pipeline framework. */
    public static final String TYPE = "muvera_query";

    /** Script param key under which the per-token query multi-vectors are passed in. */
    static final String QUERY_VECTORS_PARAM = "query_vectors";

    /** Configuration property keys. */
    static final String CONFIG_TARGET_FIELD = "target_field";
    static final String CONFIG_DIM = "dim";
    static final String CONFIG_K_SIM = "k_sim";
    static final String CONFIG_DIM_PROJ = "dim_proj";
    static final String CONFIG_R_REPS = "r_reps";
    static final String CONFIG_SEED = "seed";
    static final String CONFIG_FDE_DIMENSION = "fde_dimension";

    /** Default values for optional configuration properties. */
    static final int DEFAULT_K_SIM = 4;
    static final int DEFAULT_DIM_PROJ = 8;
    static final int DEFAULT_R_REPS = 20;
    static final long DEFAULT_SEED = 42L;

    private final String targetField;
    private final MuveraEncoder encoder;
    private final int dim;
    private final int fdeDimension;

    MuveraSearchRequestProcessor(
        String tag,
        String description,
        boolean ignoreFailure,
        String targetField,
        MuveraEncoder encoder,
        int dim,
        int fdeDimension
    ) {
        super(tag, description, ignoreFailure);
        this.targetField = targetField;
        this.encoder = encoder;
        this.dim = dim;
        this.fdeDimension = fdeDimension;
    }

    /**
     * Encodes the query multi-vectors into an FDE and exposes it via the pipeline
     * processing context so the surrounding template query can resolve
     * {@code ${target_field}} during query rewrite.
     */
    @Override
    public SearchRequest processRequest(SearchRequest request, PipelineProcessingContext requestContext) throws Exception {
        if (request.source() == null) {
            return request;
        }

        float[][] multiVectors = extractQueryVectors(request);
        if (multiVectors == null) {
            return request;
        }

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

        // The pipeline framework serializes context attributes via XContentBuilder.value(Object),
        // which natively handles float[] (no Float boxing). The template query reads this
        // attribute and substitutes it into the ${targetField} placeholder during query rewrite.
        requestContext.setAttribute(targetField, queryFde);

        return request;
    }

    /**
     * Fallback overload invoked when no {@link PipelineProcessingContext} is available.
     * Template query resolution requires the pipeline context, so without it we cannot
     * substitute the placeholder; the request is returned unchanged.
     */
    @Override
    public SearchRequest processRequest(SearchRequest request) {
        return request;
    }

    /**
     * Extracts the {@code query_vectors} parameter from a {@link TemplateQueryBuilder}
     * content map. Returns {@code null} when the query is not a template query or does
     * not contain a recognizable {@code query_vectors} payload.
     */
    private float[][] extractQueryVectors(SearchRequest request) {
        QueryBuilder query = request.source().query();
        if (query instanceof TemplateQueryBuilder == false) {
            return null;
        }
        TemplateQueryBuilder templateQuery = (TemplateQueryBuilder) query;
        Object queryVectorsObj = extractQueryVectorsFromContent(templateQuery.getContent());
        if (queryVectorsObj == null) {
            return null;
        }
        return parseMultiVectors(queryVectorsObj);
    }

    /**
     * Navigates the template content map to find {@code query_vectors} in
     * {@code script_score -> script -> params -> query_vectors}.
     */
    @SuppressWarnings("unchecked")
    private Object extractQueryVectorsFromContent(Map<String, Object> content) {
        if (content == null) {
            return null;
        }
        Object scriptScoreObj = content.get("script_score");
        if (scriptScoreObj instanceof Map == false) {
            return null;
        }
        Object scriptObj = ((Map<String, Object>) scriptScoreObj).get("script");
        if (scriptObj instanceof Map == false) {
            return null;
        }
        Object paramsObj = ((Map<String, Object>) scriptObj).get("params");
        if (paramsObj instanceof Map == false) {
            return null;
        }
        return ((Map<String, Object>) paramsObj).get(QUERY_VECTORS_PARAM);
    }

    /**
     * Parses and validates a {@code query_vectors} value into {@code float[][]}.
     */
    private float[][] parseMultiVectors(Object queryVectorsObj) {
        if (queryVectorsObj instanceof List == false) {
            throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] must be a list of vectors");
        }

        List<?> outerList = (List<?>) queryVectorsObj;
        int numTokens = outerList.size();
        if (numTokens == 0) {
            throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] must not be empty");
        }

        float[][] multiVectors = new float[numTokens][dim];
        for (int t = 0; t < numTokens; t++) {
            Object vecObj = outerList.get(t);
            if (vecObj instanceof List == false) {
                throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] element at index [" + t + "] must be a list of numbers");
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
                    throw new IllegalArgumentException("[" + QUERY_VECTORS_PARAM + "] element at [" + t + "][" + d + "] is not a number");
                }
                multiVectors[t][d] = ((Number) numObj).floatValue();
            }
        }
        return multiVectors;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    /** Factory used by the search pipeline framework to instantiate the processor. */
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
            String targetField = ConfigurationUtils.readStringProperty(TYPE, tag, config, CONFIG_TARGET_FIELD);

            // dim is required — no sensible default since it depends on the embedding model.
            Integer dimValue = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_DIM, null);
            if (dimValue == null) {
                throw ConfigurationUtils.newConfigurationException(
                    TYPE,
                    tag,
                    CONFIG_DIM,
                    "required property is missing. Set this to your embedding model's vector dimension (e.g. 128)."
                );
            }
            int dim = dimValue;

            int kSim = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_K_SIM, DEFAULT_K_SIM);
            int dimProj = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_DIM_PROJ, DEFAULT_DIM_PROJ);
            int rReps = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_R_REPS, DEFAULT_R_REPS);
            long seed = MuveraProcessorUtils.readLongProperty(TYPE, tag, config, CONFIG_SEED, DEFAULT_SEED);

            MuveraEncoder encoder = new MuveraEncoder(dim, kSim, dimProj, rReps, seed);
            int computedDimension = encoder.getEmbeddingSize();

            // Validate fde_dimension if provided so users catch mapping mismatches at
            // pipeline creation rather than at search time.
            Integer userDimension = ConfigurationUtils.readIntProperty(TYPE, tag, config, CONFIG_FDE_DIMENSION, null);
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

            // Surface the computed FDE dimension via the processor description (visible
            // through GET _search/pipeline) instead of a cluster-level log line, since
            // cluster logs aren't end-user visible.
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

            return new MuveraSearchRequestProcessor(tag, description, ignoreFailure, targetField, encoder, dim, computedDimension);
        }
    }
}
