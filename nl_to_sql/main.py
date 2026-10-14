import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize


AGGREGATE_MAP = {
    "count": "COUNT",
    "total": "COUNT",
    "average": "AVG",
    "avg": "AVG",
    "sum": "SUM",
    "maximum": "MAX",
    "max": "MAX",
    "minimum": "MIN",
    "min": "MIN",
}

OPERATORS = {
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "greater than": ">",
    "less than": "<",
    "more than": ">",
    "equal to": "=",
    "equals": "=",
    "is": "=",
    "contains": "LIKE",
}


@dataclass
class ParsedIntent:
    table: str
    columns: List[str]
    where_clause: Optional[str]
    limit: Optional[int]


class NLToSQLConverter:
    def __init__(self, schema: Dict[str, List[str]]) -> None:
        self.schema = schema
        self.stemmer = PorterStemmer()
        self.all_columns: Dict[str, Set[str]] = {
            table: {col.lower() for col in cols} for table, cols in schema.items()
        }

    def to_sql(self, text: str) -> str:
        normalized = self._normalize_text(text)
        table = self._infer_table(normalized)
        columns = self._infer_columns(normalized, table)
        where_clause = self._infer_where(normalized, table)
        limit = self._infer_limit(normalized)
        intent = ParsedIntent(table=table, columns=columns, where_clause=where_clause, limit=limit)
        return self._build_query(intent)

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in wordpunct_tokenize(text) if token.strip()]

    def _stem(self, token: str) -> str:
        return self.stemmer.stem(token)

    def _infer_table(self, text: str) -> str:
        tokens = self._tokenize(text)
        token_stems = {self._stem(token) for token in tokens}
        candidates: List[Tuple[int, str]] = []

        for table in self.schema:
            score = 0
            table_tokens = table.replace("_", " ").split()
            if table in text:
                score += 4
            if table.rstrip("s") in text:
                score += 2
            for table_token in table_tokens:
                if table_token in tokens:
                    score += 1
                if self._stem(table_token) in token_stems:
                    score += 1
            if score > 0:
                candidates.append((score, table))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

        return next(iter(self.schema))

    def _infer_columns(self, text: str, table: str) -> List[str]:
        table_columns = self.schema[table]
        if "all" in text or "everything" in text or "*" in text:
            return ["*"]

        for key, sql_fn in AGGREGATE_MAP.items():
            if key in text:
                col = self._best_column_match(text, table, excluded={"id"})
                if col:
                    return [f"{sql_fn}({col})"]
                return [f"{sql_fn}(*)"]

        matches = []
        for col in table_columns:
            col_phrase = col.replace("_", " ")
            if col in text or col_phrase in text:
                matches.append(col)

        if matches:
            return sorted(set(matches), key=matches.index)

        explicit_select = re.search(r"(show|select|get|list)\s+(.*?)\s+(from|in)\s+", text)
        if explicit_select:
            requested = explicit_select.group(2)
            possible = [part.strip() for part in re.split(r",| and ", requested) if part.strip()]
            resolved = [self._best_column_match(chunk, table) for chunk in possible]
            resolved = [item for item in resolved if item]
            if resolved:
                return resolved

        if "how many" in text:
            return ["COUNT(*)"]

        return ["*"]

    def _best_column_match(self, text: str, table: str, excluded: Optional[Set[str]] = None) -> Optional[str]:
        excluded = excluded or set()
        text_tokens = set(self._tokenize(text))
        text_stems = {self._stem(token) for token in text_tokens}

        scored: List[Tuple[int, str]] = []
        for column in self.schema[table]:
            if column in excluded:
                continue
            score = 0
            col_tokens = column.replace("_", " ").split()
            if column in text:
                score += 4
            if column.replace("_", " ") in text:
                score += 3
            for col_token in col_tokens:
                if col_token in text_tokens:
                    score += 2
                if self._stem(col_token) in text_stems:
                    score += 1
            if score > 0:
                scored.append((score, column))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    def _infer_where(self, text: str, table: str) -> Optional[str]:
        lower_text = text

        for phrase, operator in sorted(OPERATORS.items(), key=lambda item: len(item[0]), reverse=True):
            if phrase not in lower_text:
                continue

            parts = lower_text.split(phrase, maxsplit=1)
            if len(parts) < 2:
                continue

            left, right = parts[0], parts[1]
            column = self._best_column_match(left, table)
            if not column:
                continue

            value = self._extract_value(right)
            if value is None:
                continue

            if operator == "LIKE":
                return f"{column} LIKE '%{value}%'"
            if self._is_number(value):
                return f"{column} {operator} {value}"
            return f"{column} {operator} '{value}'"

        direct_match = re.search(r"where\s+([a-zA-Z_ ]+)\s*=\s*([a-zA-Z0-9_@.\-]+)", lower_text)
        if direct_match:
            column = self._best_column_match(direct_match.group(1), table)
            value = direct_match.group(2)
            if column:
                if self._is_number(value):
                    return f"{column} = {value}"
                return f"{column} = '{value}'"

        return None

    def _extract_value(self, text: str) -> Optional[str]:
        value = text.strip()
        value = re.sub(r"^(to|than|is|the)\s+", "", value)
        value = re.sub(r"\s+(rows|records|users|orders|products)$", "", value)
        value = value.strip(" '\".,")
        if not value:
            return None
        return value

    def _infer_limit(self, text: str) -> Optional[int]:
        limit_match = re.search(r"(top|first|limit)\s+(\d+)", text)
        if limit_match:
            return int(limit_match.group(2))
        return None

    def _is_number(self, value: str) -> bool:
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", value))

    def _build_query(self, intent: ParsedIntent) -> str:
        column_sql = ", ".join(intent.columns)
        query = f"SELECT {column_sql} FROM {intent.table}"
        if intent.where_clause:
            query += f" WHERE {intent.where_clause}"
        if intent.limit is not None:
            query += f" LIMIT {intent.limit}"
        return query + ";"


def load_schema(schema_path: Path) -> Dict[str, List[str]]:
    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert natural language text into SQL.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("schema.json"),
        help="Path to JSON schema file (default: schema.json next to script).",
    )
    parser.add_argument("text", nargs="+", help="Natural language query.")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    converter = NLToSQLConverter(schema)
    input_text = " ".join(args.text)
    sql_query = converter.to_sql(input_text)
    print(sql_query)


if __name__ == "__main__":
    main()
