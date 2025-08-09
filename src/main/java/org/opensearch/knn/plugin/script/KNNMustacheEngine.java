/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import com.github.mustachejava.DefaultMustacheFactory;
import com.github.mustachejava.Mustache;
import com.github.mustachejava.MustacheException;
import org.opensearch.script.GeneralScriptException;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.opensearch.script.ScriptException;
import org.opensearch.script.TemplateScript;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.StringReader;
import java.io.StringWriter;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * Mustache script engine for k-NN query templating.
 */
public class KNNMustacheEngine implements ScriptEngine {

    public static final String NAME = "knn-mustache";
    private static final Logger logger = LogManager.getLogger(KNNMustacheEngine.class);

    @Override
    public String getType() {
        return NAME;
    }

    @Override
    public <T> T compile(String templateName, String templateSource, ScriptContext<T> context, Map<String, String> options) {
        KNNCounter.SCRIPT_COMPILATIONS.increment();

        if (!context.instanceClazz.equals(TemplateScript.class)) {
            KNNCounter.SCRIPT_COMPILATION_ERRORS.increment();
            throw new IllegalArgumentException("knn-mustache engine only supports TemplateScript context, got [" + context.name + "]");
        }

        try (StringReader reader = new StringReader(templateSource)) {
            DefaultMustacheFactory factory = new DefaultMustacheFactory();
            Mustache template = factory.compile(reader, templateName != null ? templateName : "knn-template");

            TemplateScript.Factory compiled = params -> new KNNMustacheTemplateScript(template, params);
            return context.factoryClazz.cast(compiled);
        } catch (MustacheException ex) {
            KNNCounter.SCRIPT_COMPILATION_ERRORS.increment();
            logger.error("Failed to compile k-NN Mustache template [{}]: {}", templateName, ex.getMessage());
            throw new ScriptException(ex.getMessage(), ex, Collections.emptyList(), templateSource, NAME);
        }
    }

    @Override
    public Set<ScriptContext<?>> getSupportedContexts() {
        return Collections.singleton(TemplateScript.CONTEXT);
    }

    /**
     * Template script implementation for k-NN query generation.
     */
    private static class KNNMustacheTemplateScript extends TemplateScript {
        private final Mustache template;
        private final Map<String, Object> params;

        KNNMustacheTemplateScript(Mustache template, Map<String, Object> params) {
            super(params);
            this.template = template;
            this.params = params;
        }

        @Override
        public String execute() {
            StringWriter writer = new StringWriter();
            try {
                template.execute(writer, params);
                return writer.toString();
            } catch (Exception e) {
                logger.error("Error executing k-NN mustache template: {}", e.getMessage());
                throw new GeneralScriptException("Error executing k-NN mustache template: " + e.getMessage(), e);
            }
        }
    }
}
