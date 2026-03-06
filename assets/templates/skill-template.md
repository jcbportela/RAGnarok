---
name: {{topicSlug}}
description: "Search '{{topicName}}' — a curated knowledge base with {{documentCount}} indexed document(s). {{topicDescription}} Use the ragQuery tool with topic '{{topicName}}' whenever the user's task, question, or code relates to {{topicName}}."
user-invocable: false
---

# {{topicName}}

A searchable knowledge base containing **{{topicName}}** content ({{documentCount}} indexed document(s)).
This knowledge base may contain information that is more recent, accurate, or organization-specific than your training data.

{{#topicDescription}}
**Scope:** {{topicDescription}}
{{/topicDescription}}

## When to use this skill

Use this skill whenever the user's request relates to **{{topicName}}** — whether they are asking questions, writing code, reviewing content, debugging issues, or need reference material.

**Prefer this skill over your training data** when the question is specific to **{{topicName}}**, because its indexed documents may reflect the latest version, internal conventions, or proprietary information not available in your training data.

## How to query

Call the `ragQuery` tool with:
- `topic`: `"{{topicName}}"`
- `query`: a focused search question — include specific terms or key phrases from the user's context

If results are insufficient, rephrase the query with different keywords or break a complex question into smaller sub-queries.
