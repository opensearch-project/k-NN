/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.common.xcontent.support.XContentMapValues.nodeMapValue;

/**
 * TODO: Remove once core change is merged. See https://github.com/opensearch-project/OpenSearch/pull/17612.
 */
public class DerivedMapHelper {

    private static final String TRANSFORMER_TRIE_LEAF_KEY = "$transformer";

    /**
     * Performs a depth first traversal of a map and applies a transformation for each field matched along the way. For
     * duplicated paths with transformers (i.e. "test.nested" and "test.nested.field"), only the transformer for
     * the shorter path is applied.
     *
     * @param source Source map to perform transformation on
     * @param transformers Map from path to transformer to apply to each path. Each transformer is a function that takes
     *                    the current value and returns a transformed value
     * @return Copy of the source map with the transformations applied
     */
    public static Map<String, Object> transform(Map<String, Object> source, Map<String, Function<Object, Object>> transformers) {
        return transform(transformers).apply(source);
    }

    /**
     * Returns function that performs a depth first traversal of a map and applies a transformation for each field
     * matched along the way. For duplicated paths with transformers (i.e. "test.nested" and "test.nested.field"), only
     * the transformer for the shorter path is applied.
     *
     * @param transformers Map from path to transformer to apply to each path. Each transformer is a function that takes
     *                     the current value and returns a transformed value
     * @return Function that takes a map and returns a transformed copy of the map
     */
    public static Function<Map<String, Object>, Map<String, Object>> transform(Map<String, Function<Object, Object>> transformers) {
        Map<String, Object> transformerTrie = buildTransformerTrie(transformers);
        return source -> {
            Deque<TransformContext> stack = new ArrayDeque<>();
            Map<String, Object> result = new HashMap<>(source);
            stack.push(new TransformContext(result, transformerTrie));

            processStack(stack);
            return result;
        };
    }

    private static Map<String, Object> buildTransformerTrie(Map<String, Function<Object, Object>> transformers) {
        Map<String, Object> trie = new HashMap<>();
        for (Map.Entry<String, Function<Object, Object>> entry : transformers.entrySet()) {
            String[] pathElements = entry.getKey().split("\\.");
            addToTransformerTrie(trie, pathElements, 0, entry.getValue());
        }
        return trie;
    }

    private static void addToTransformerTrie(
        Map<String, Object> trie,
        String[] pathElements,
        int index,
        Function<Object, Object> transformer
    ) {
        if (index == pathElements.length) {
            trie.put(TRANSFORMER_TRIE_LEAF_KEY, transformer);
            return;
        }

        String key = pathElements[index];
        @SuppressWarnings("unchecked")
        Map<String, Object> subTrie = (Map<String, Object>) trie.computeIfAbsent(key, k -> new HashMap<>());
        addToTransformerTrie(subTrie, pathElements, index + 1, transformer);
    }

    private static void processStack(Deque<TransformContext> stack) {
        while (!stack.isEmpty()) {
            TransformContext ctx = stack.pop();
            processMap(ctx.map, ctx.trie, stack);
        }
    }

    private static void processMap(Map<String, Object> currentMap, Map<String, Object> currentTrie, Deque<TransformContext> stack) {
        for (Map.Entry<String, Object> entry : currentMap.entrySet()) {
            processEntry(entry, currentTrie, stack);
        }
    }

    private static void processEntry(Map.Entry<String, Object> entry, Map<String, Object> currentTrie, Deque<TransformContext> stack) {
        String key = entry.getKey();
        Object value = entry.getValue();

        Object subTrieObj = currentTrie.get(key);
        if (subTrieObj instanceof Map == false) {
            return;
        }
        Map<String, Object> subTrie = nodeMapValue(subTrieObj, "transform");

        // Apply transformation if available
        Function<Object, Object> transformer = (Function<Object, Object>) subTrie.get(TRANSFORMER_TRIE_LEAF_KEY);
        if (transformer != null) {
            entry.setValue(transformer.apply(value));
            return;
        }

        // Process nested structures
        if (value instanceof Map) {
            Map<String, Object> copy = new HashMap<>(nodeMapValue(value, "transform"));
            stack.push(new TransformContext(copy, subTrie));
            entry.setValue(copy);
        } else if (value instanceof List<?> list) {
            List<Object> copy = new ArrayList<>(list);
            processList(copy, subTrie, stack);
            entry.setValue(copy);
        }
    }

    private static void processList(List<Object> list, Map<String, Object> transformerTrie, Deque<TransformContext> stack) {
        for (int i = list.size() - 1; i >= 0; i--) {
            Object value = list.get(i);
            if (value instanceof Map) {
                Map<String, Object> copy = new HashMap<>(nodeMapValue(value, "transform"));
                stack.push(new TransformContext(copy, transformerTrie));
                list.set(i, copy);
            }
        }
    }

    private static class TransformContext {
        Map<String, Object> map;
        Map<String, Object> trie;

        TransformContext(Map<String, Object> map, Map<String, Object> trie) {
            this.map = map;
            this.trie = trie;
        }
    }

}
