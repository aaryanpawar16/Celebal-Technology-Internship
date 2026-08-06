# Single Agent Pipeline Project

A single-agent smart assistant that routes user queries to the appropriate tool based on intent and returns structured JSON output.

## Features

- **Intent-based routing** — inspects the query and dispatches to the right handler
- **Calculator tool** — evaluates math expressions
- **Keyword extractor tool** — pulls keywords (words longer than 4 chars) from text
- **General fallback** — returns a direct response for anything else
- **Error handling** — invalid expressions or unexpected failures return a structured error

## Files

- `week_8_assignment.ipynb` — the notebook containing the tools, agent logic, test cases, and an interactive loop

## Agent Routing Logic

| Query contains | Route          | Tool used          |
|-----------------|----------------|---------------------|
| `"calculate"`   | `calculation`  | `calculator()`      |
| `"keywords"`    | `keywords`     | `extract_keywords()`|
| anything else   | `general`      | direct response      |
| exception raised| `error`        | —                    |

## Output Format

```json
{
  "type": "calculation / keywords / general / error",
  "result": ...
}
```

## Usage

Run all cells, then either:

1. Check the pre-built test cases printed automatically, or
2. Use the interactive cell — type a query, or `exit` to stop.

### Example Queries

- `Calculate 45 * 3 - 10`
- `Calculate 100 / 0` (error case)
- `Extract keywords from Deep learning models require large datasets`
- `What is the capital of France?` (general fallback)
