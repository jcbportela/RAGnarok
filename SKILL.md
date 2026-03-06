---
name: query-rag
description: Query RAG databases for documentation. Use when the user asks about topics present in RAG.
---

# RAG Query Skill (Agent Author Guide)

Use this skill to retrieve information from indexed documentation via `ragQuery`.
This guide is optimized for **correct topic selection, fast retrieval, and reliable attribution**.

## When to Use
- The user asks for information that should exist in a known RAG topic.
- The user needs grounded answers from indexed docs, not generic model knowledge.
- The question benefits from retrieval strategy selection (`hybrid`, `bm25`, `vector`, `ensemble`).

## Required Inputs and Runtime Constraints
- `topic` (required): topic name to search.
- `query` (required): retrieval question.
- `retrievalStrategy` (optional): `hybrid` | `bm25` | `vector` | `ensemble`.
- `topK` (optional): number of results; runtime accepts integer range `1..20`.


## Topic Matching Rules (Must Follow)
The tool performs topic matching automatically:
1. Exact topic match first.
2. If no exact match, use the best semantic match.
3. If only one topic exists, it may be used as fallback.

After every query:
- Check `topicMatched` and compare it to requested `topic`.
- If they differ, proceed but **explicitly disclose** the mismatch in the response.
- If the mismatch could change correctness (different product/domain), ask user confirmation before final conclusions.

Required mismatch disclosure format:
"Requested topic: `<requestedTopic>`. Matched topic: `<topicMatched>`."

If no usable match is returned:
- State that the requested topic was not found.
- Ask the user to confirm topic name or provide an alternative.
- Include available topics if present.

## Most Efficient Query Flow
1. Start with exact topic and focused query.
2. Use default behavior first (no manual overrides) unless query type clearly requires tuning.
3. Choose strategy by query type:
   - `hybrid` (default best general choice): mixed semantic + keyword.
   - `bm25`: exact identifiers, error codes, flags, command names, API symbols.
   - `vector`: conceptual/semantic questions where exact words may differ.
   - `ensemble`: hardest queries needing higher recall (more expensive/slower).
4. RAGnarok always uses agentic mode with query planning:
   - comparisons across sections/documents,
   - synthesis of several constraints,
   - stepwise troubleshooting requiring decomposition.
5. Keep `topK` small by default; increase only when evidence is insufficient.

> Note that except for `topic` and `query`, all parameters are optional and should be used strategically based on the question type and initial results.

## Query Patterns

### Fast default
```ts
ragQuery(topic="<topic>", query="<question>")
```

### Exact-term heavy
```ts
ragQuery(
  topic="<topic>",
  query="<question_with_identifiers>",
  retrievalStrategy="bm25"
)
```

### Complex synthesis
```ts
ragQuery(
  topic="<topic>",
  query="<multi_part_question>",
  retrievalStrategy="hybrid"
)
```

## Output Requirements (Always)
- Provide answer grounded in retrieved chunks only.
- Include topic match transparency when applicable.
- Always include source attribution from returned metadata.

Preferred source format:
- `Source: <documentName> (section: <sectionTitle or heading>)`
- If section is unavailable: `Source: <documentName>`
- If chunk metadata is useful: include chunk position (`chunkIndex` or line/char span if available).

Do not require page numbers unless page metadata exists.

## Error and Empty-Result Handling
- If tool returns no chunks/results:
  - say no relevant evidence was found,
  - suggest a tighter query, different strategy, or different topic,
  - offer to rerun with `bm25` (for exact terms) or `ensemble` (for recall).
- If topic/doc store is missing:
  - tell user the topic may not be created or indexed,
  - ask for the intended topic and offer to guide ingestion/indexing workflow.

## Response Templates

### Normal grounded response
"<Answer>. Source: <documentName> (section: <section>)."

### Topic mismatch disclosure
"Requested topic: `<requestedTopic>`. Matched topic: `<topicMatched>`. Proceeding with matched topic results."

### Topic not found
"I could not find the requested RAG topic `<requestedTopic>`. Matched topic: `<topicMatched or none>`. Available topics: <list if available>. Please confirm the topic to search."

### No evidence found
"I couldn't find enough relevant evidence for this query in topic `<topicMatched or requestedTopic>`. I can retry with `bm25` (exact terms) or `ensemble` (higher recall)."

## Best Practices Checklist
- Use exact topic names when possible.
- Verify `topicMatched` before final answer.
- Prefer `hybrid` unless query characteristics suggest otherwise.
- Reserve agentic mode for truly multi-step questions.
- Keep answers concise and cite sources from returned metadata.