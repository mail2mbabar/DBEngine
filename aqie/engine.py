import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


DATE_KEYWORDS = ["date", "time", "year", "month", "day", "timestamp", "dob", "birth", "created", "updated"]
ID_KEYWORDS = ["id", "code", "number", "num", "no", "ref", "key", "index", "uuid"]
MONEY_KEYWORDS = ["price", "amount", "cost", "salary", "revenue", "fee", "balance", "payment", "income"]
NAME_KEYWORDS = ["name", "title", "label", "description", "product", "category"]
SCORE_KEYWORDS = ["score", "grade", "gpa", "rating", "result", "mark", "rank", "performance"]
GEO_KEYWORDS = ["country", "city", "state", "region", "location", "address", "zip", "lat", "lon", "geo"]
DOMAIN_HINTS = {
    "healthcare": ["patient", "patients", "blood", "group", "diagnosis", "hospital", "admission", "discharge"],
    "finance": ["transaction", "payment", "fraud", "account", "balance", "card", "bank"],
    "sales": ["sales", "revenue", "product", "order", "customer", "unit", "price"],
    "education": ["student", "grade", "gpa", "marks", "course", "school"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _kw(col_name: str, kw_list: List[str]) -> int:
    c = col_name.lower().replace("_", " ")
    return int(any(k in c for k in kw_list))


def _normalize_token_variants(text: str) -> str:
    t = text.lower()
    replacements = {
        "solds": "sold",
        "qty": "quantity",
        "unit solds": "units sold",
        "bloodgroup": "blood group",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t


def _clean_column_phrase(text: str) -> str:
    t = _normalize_token_variants(text.lower())
    t = re.sub(r"\b(only|column|columns|data|dataset|table|which|that|has|having|with)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _clean_filter_value(raw_col: str, raw_val: str) -> str:
    col = _normalize_token_variants(raw_col.lower())
    val = _normalize_token_variants(raw_val.lower()).strip()
    val = re.sub(r"^[\"']|[\"']$", "", val).strip()
    val = re.sub(r"\bonly\b$", "", val).strip()
    val = re.sub(r"\s+", " ", val).strip()
    if not val:
        return raw_val.strip()

    col_tokens = _tokenize(col)
    val_tokens = _tokenize(val)
    # Remove duplicated leading column words in value, e.g. "provider blue cross".
    for n in range(min(3, len(val_tokens)), 0, -1):
        lead = " ".join(val_tokens[:n])
        if lead in col:
            val_tokens = val_tokens[n:]
            break
    cleaned = " ".join(val_tokens).strip()
    return cleaned if cleaned else val


STOPWORDS = {
    "give",
    "me",
    "data",
    "which",
    "that",
    "has",
    "have",
    "with",
    "show",
    "rows",
    "row",
    "record",
    "records",
    "only",
    "please",
    "need",
    "want",
    "search",
    "find",
}


@dataclass
class QueryResult:
    sql: str
    explanation: str
    dataframe: pd.DataFrame
    warnings: List[str]


class AQIEEngine:
    def __init__(self) -> None:
        self.conn = duckdb.connect(database=":memory:")
        self.tables: Dict[str, pd.DataFrame] = {}
        self.schema_df: pd.DataFrame = pd.DataFrame()
        self.feature_df: pd.DataFrame = pd.DataFrame()
        self.rf_model: Optional[RandomForestClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.llm_enabled: bool = False
        self.llm_model: str = "gpt-4o-mini"
        self.llm_base_url: Optional[str] = None
        self._llm_client: Optional[OpenAI] = None
        self._value_cache: Dict[str, Dict[str, List[str]]] = {}
        self.last_llm_status: str = ""
        self.last_filter_mode: str = "exact"

    def configure_llm(
        self,
        enabled: bool,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ) -> None:
        self.llm_enabled = enabled and bool(api_key)
        self.llm_model = model.strip() if model else "gpt-4o-mini"
        self.llm_base_url = base_url.strip() if base_url else None
        self._llm_client = None
        if self.llm_enabled:
            kwargs = {"api_key": api_key}
            if self.llm_base_url:
                kwargs["base_url"] = self.llm_base_url
            self._llm_client = OpenAI(**kwargs)

    def load_files(self, uploaded_files) -> None:
        self.tables = {}
        self.conn = duckdb.connect(database=":memory:")
        for up in uploaded_files:
            name = up.name.rsplit(".", 1)[0]
            table_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                xls = pd.ExcelFile(up)
                if len(xls.sheet_names) == 1:
                    df = pd.read_excel(up, sheet_name=xls.sheet_names[0])
                    self.tables[table_name] = df
                    self.conn.register(table_name, df)
                    continue
                for sh in xls.sheet_names:
                    df = pd.read_excel(up, sheet_name=sh)
                    tname = f"{table_name}_{re.sub(r'[^a-zA-Z0-9_]', '_', sh).lower()}"
                    self.tables[tname] = df
                    self.conn.register(tname, df)
                continue

            self.tables[table_name] = df
            self.conn.register(table_name, df)

        self.schema_df = self.discover_schema()
        self.feature_df = self.extract_features()
        self._train_semantic_model()
        self._predict_semantic_types()
        self._build_value_cache()

    def discover_schema(self) -> pd.DataFrame:
        rows = []
        for tname, df in self.tables.items():
            for col in df.columns:
                s = df[col]
                n_total = len(s) if len(s) else 1
                n_null = int(s.isna().sum())
                n_unique = int(s.nunique(dropna=True))
                cardinality = n_unique / n_total
                if pd.api.types.is_numeric_dtype(s):
                    structural = "Numeric"
                elif pd.api.types.is_datetime64_any_dtype(s):
                    structural = "Datetime"
                else:
                    structural = "Categorical"
                is_pk = n_unique == len(df) and n_null == 0
                rows.append(
                    {
                        "table": tname,
                        "column": col,
                        "dtype": str(s.dtype),
                        "structural_type": structural,
                        "missing": n_null,
                        "missing_pct": round(100 * n_null / n_total, 3),
                        "unique": n_unique,
                        "cardinality_ratio": round(cardinality, 4),
                        "pk_candidate": "Yes" if is_pk else "No",
                    }
                )
        return pd.DataFrame(rows)

    def extract_features(self) -> pd.DataFrame:
        rows = []
        for tname, df in self.tables.items():
            for col in df.columns:
                s = df[col]
                s_non = s.dropna()
                total = max(len(s), 1)
                non = max(len(s_non), 1)

                prop_numeric = pd.to_numeric(s_non, errors="coerce").notna().mean() if len(s_non) else 0.0
                prop_date = pd.to_datetime(s_non, errors="coerce").notna().mean() if len(s_non) else 0.0
                card = s_non.nunique() / total if len(s_non) else 0.0

                text_vals = s_non.astype(str)
                token_lens = text_vals.str.split().map(len) if len(text_vals) else pd.Series([0])
                mean_token_len = float(token_lens.mean())
                std_token_len = float(token_lens.std()) if len(token_lens) > 1 else 0.0

                has_currency = int(text_vals.str.contains(r"[$€£]|usd|pkr|inr|eur", case=False, regex=True).any()) if len(text_vals) else 0
                has_percent = int(text_vals.str.contains(r"%", regex=True).any()) if len(text_vals) else 0

                as_num = pd.to_numeric(s_non, errors="coerce").dropna()
                if len(as_num) > 1 and as_num.mean() != 0:
                    coeff_var = float(as_num.std() / as_num.mean())
                else:
                    coeff_var = 0.0

                probs = text_vals.value_counts(normalize=True)
                entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if len(probs) else 0.0
                is_binary = int(s_non.nunique() == 2)

                rows.append(
                    {
                        "table": tname,
                        "column": col,
                        "prop_numeric": round(prop_numeric, 4),
                        "prop_date": round(prop_date, 4),
                        "cardinality_ratio": round(card, 4),
                        "mean_token_len": round(mean_token_len, 4),
                        "std_token_len": round(std_token_len, 4),
                        "has_currency": has_currency,
                        "has_percent": has_percent,
                        "coeff_variation": round(coeff_var, 4),
                        "value_entropy": round(entropy, 4),
                        "is_binary": is_binary,
                        "kw_date": _kw(col, DATE_KEYWORDS),
                        "kw_id": _kw(col, ID_KEYWORDS),
                        "kw_money": _kw(col, MONEY_KEYWORDS),
                        "kw_name": _kw(col, NAME_KEYWORDS),
                        "kw_score": _kw(col, SCORE_KEYWORDS),
                        "kw_geo": _kw(col, GEO_KEYWORDS),
                    }
                )
        return pd.DataFrame(rows)

    def _heuristic_label(self, row: pd.Series) -> str:
        if row["kw_id"] == 1 and row["cardinality_ratio"] > 0.8:
            return "Identifier"
        if row["kw_date"] == 1 or row["prop_date"] > 0.6:
            return "DateTime"
        if row["kw_money"] == 1:
            return "Monetary"
        if row["kw_score"] == 1:
            return "Score"
        if row["kw_geo"] == 1:
            return "Location"
        if row["prop_numeric"] > 0.8:
            return "Numeric"
        if row["is_binary"] == 1:
            return "Binary"
        if row["kw_name"] == 1:
            return "NameText"
        return "Categorical"

    def _train_semantic_model(self) -> None:
        if self.feature_df.empty:
            return
        feat_cols = [
            "prop_numeric",
            "prop_date",
            "cardinality_ratio",
            "mean_token_len",
            "std_token_len",
            "has_currency",
            "has_percent",
            "coeff_variation",
            "value_entropy",
            "is_binary",
            "kw_date",
            "kw_id",
            "kw_money",
            "kw_name",
            "kw_score",
            "kw_geo",
        ]
        labels = self.feature_df.apply(self._heuristic_label, axis=1)
        x = self.feature_df[feat_cols].fillna(0).values
        le = LabelEncoder()
        y = le.fit_transform(labels)
        model = RandomForestClassifier(n_estimators=220, random_state=42)
        model.fit(x, y)
        self.rf_model = model
        self.label_encoder = le

    def _predict_semantic_types(self) -> None:
        if self.feature_df.empty or self.rf_model is None or self.label_encoder is None:
            return
        feat_cols = [
            "prop_numeric",
            "prop_date",
            "cardinality_ratio",
            "mean_token_len",
            "std_token_len",
            "has_currency",
            "has_percent",
            "coeff_variation",
            "value_entropy",
            "is_binary",
            "kw_date",
            "kw_id",
            "kw_money",
            "kw_name",
            "kw_score",
            "kw_geo",
        ]
        x = self.feature_df[feat_cols].fillna(0).values
        y_pred = self.rf_model.predict(x)
        self.feature_df["semantic_type"] = self.label_encoder.inverse_transform(y_pred)

    def _all_table_names(self) -> List[str]:
        return list(self.tables.keys())

    def _all_columns(self) -> List[Tuple[str, str]]:
        pairs = []
        for tname, df in self.tables.items():
            for c in df.columns:
                pairs.append((tname, c))
        return pairs

    def _build_value_cache(self) -> None:
        self._value_cache = {}
        for tname, df in self.tables.items():
            self._value_cache[tname] = {}
            for col in df.columns:
                s = df[col].dropna()
                if s.empty:
                    continue
                # Keep representative unique values to support value-aware filtering.
                vals = s.astype(str).str.strip()
                uniq = vals[vals != ""].drop_duplicates().head(200).tolist()
                self._value_cache[tname][col] = uniq

    def _best_table_match(self, q: str) -> Optional[str]:
        names = self._all_table_names()
        if not names:
            return None
        toks = set(_tokenize(q))
        scores = []
        for n in names:
            nt = set(_tokenize(n))
            score = len(toks & nt)
            for domain, hints in DOMAIN_HINTS.items():
                if domain in n.lower():
                    score += sum(1 for h in hints if h in toks)
            scores.append((score, n))
        scores.sort(reverse=True)
        if scores and scores[0][0] > 0:
            return scores[0][1]
        # If table names are not helpful, score using column names.
        best_table = names[0]
        best_score = -1
        for t in names:
            col_tokens = set()
            for c in self.tables[t].columns:
                col_tokens.update(_tokenize(c))
            score = len(toks & col_tokens)
            if score > best_score:
                best_score = score
                best_table = t
        return best_table

    def _best_column_match(self, table: str, q: str, numeric_only: bool = False) -> Optional[str]:
        if table not in self.tables:
            return None
        cols = list(self.tables[table].columns)
        if numeric_only:
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.tables[table][c])]
            if num_cols:
                cols = num_cols
        q = _normalize_token_variants(q)
        q_clean = " ".join(_tokenize(q))
        map_lower = {c.lower(): c for c in cols}
        direct = get_close_matches(q_clean, map_lower.keys(), n=1, cutoff=0.4)
        if direct:
            return map_lower[direct[0]]

        q_tokens = set(_tokenize(q))
        best = None
        best_score = -1
        for c in cols:
            c_tokens = set(_tokenize(c))
            score = len(q_tokens & c_tokens)
            if any(tok in c.lower() for tok in q_tokens):
                score += 1
            if score > best_score:
                best_score = score
                best = c
        return best

    def _best_column_for_value(self, table: str, raw_value: str) -> Optional[str]:
        if table not in self._value_cache:
            return None
        target = raw_value.strip().lower()
        if not target:
            return None
        best_col = None
        best_score = 0.0
        for col, values in self._value_cache[table].items():
            for v in values:
                lv = v.lower()
                if lv == target:
                    return col
                if target in lv or lv in target:
                    score = min(len(target), len(lv)) / max(len(target), len(lv))
                    if score > best_score:
                        best_score = score
                        best_col = col
        return best_col if best_score >= 0.5 else None

    def _extract_condition_value(self, q: str) -> Optional[Tuple[str, str]]:
        q = _normalize_token_variants(q)
        # Strong pattern for blood group style prompts.
        m = re.search(r"(?:has|with|where|whose)(?:\s+only)?\s+([a-zA-Z0-9_ ]*blood[ a-zA-Z0-9_]*)\s+((?:ab|a|b|o)[+-])(?=\s|$|[.,;:])", q)
        if m:
            return _clean_column_phrase(m.group(1)), m.group(2)
        patterns = [
            r"(?:where|with|for)\s+([a-zA-Z0-9_ ]+?)\s*(?:=|\bis\b|\bequals\b)\s*['\"]?([a-zA-Z0-9_ .+\-]+)['\"]?$",
            r"([a-zA-Z0-9_ ]+?)\s*(?:=|\bis\b|\bequals\b)\s*['\"]?([a-zA-Z0-9_ .+\-]+)['\"]?$",
            r"(?:whose|where)\s+([a-zA-Z0-9_ ]+?)\s+(?:\bis\b|\bare\b)\s+['\"]?([a-zA-Z0-9_ .+\-]+)['\"]?$",
            r"(?:has|having)\s+([a-zA-Z0-9_ ]+?)\s+['\"]?([a-zA-Z0-9_ .+\-]+)['\"]?$",
            r"(?:which\s+has)\s+([a-zA-Z0-9_ ]+?)\s+['\"]?([a-zA-Z0-9_ .+\-]+)['\"]?$",
        ]
        for p in patterns:
            m = re.search(p, q.lower())
            if m:
                raw_col = _clean_column_phrase(m.group(1))
                raw_val = m.group(2).strip()
                # Avoid false condition parsing for projection asks like
                # "has name and blood type column".
                if (" and " in raw_val.lower()) and ("column" in raw_val.lower()):
                    return None
                return raw_col, raw_val
        # Special handling for compact categorical prompts like "blood type b-".
        blood_val = re.search(r"\b(ab|a|b|o)[+-](?=\s|$|[.,;:])", q.lower())
        if blood_val and ("blood" in q.lower()):
            return "blood type", blood_val.group(0)
        return None

    def _extract_contains_value(self, q: str) -> Optional[str]:
        qn = _normalize_token_variants(q.lower())
        patterns = [
            r"\bcontains\s+([a-zA-Z0-9_ .+\-]+)$",
            r"\bwith\s+([a-zA-Z0-9_ .+\-]+)$",
        ]
        for p in patterns:
            m = re.search(p, qn)
            if m:
                val = m.group(1).strip()
                val = re.sub(r"\bonly\b$", "", val).strip()
                return val if len(val) >= 2 else None
        return None

    def _extract_value_phrase(self, q: str) -> Optional[str]:
        qn = _normalize_token_variants(q.lower())
        patterns = [
            r"\bcontains\s+([a-zA-Z0-9_ .+\-]+)$",
            r"\b(?:where|with|for|has|having)\s+[a-zA-Z0-9_ ]+?\s*(?:=|\bis\b|\bequals\b)\s*([a-zA-Z0-9_ .+\-]+)$",
            r"\b(?:has|with)\s+([a-zA-Z0-9_ .+\-]+)$",
        ]
        for p in patterns:
            m = re.search(p, qn)
            if m:
                val = m.group(1).strip()
                val = re.sub(r"\bonly\b$", "", val).strip()
                if len(val) >= 2:
                    return val
        return None

    def _text_columns(self, table: str) -> List[str]:
        if table not in self.tables:
            return []
        cols = []
        df = self.tables[table]
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols

    def _build_contains_search_sql(self, table: str, value_phrase: str, limit_n: int = 100) -> Optional[str]:
        text_cols = self._text_columns(table)
        if not text_cols:
            return None
        safe_phrase = value_phrase.replace("'", "''")
        like_clauses = [f'LOWER(CAST("{c}" AS VARCHAR)) LIKE LOWER(\'%{safe_phrase}%\')' for c in text_cols]
        where_sql = " OR ".join(like_clauses)
        return f'SELECT * FROM "{table}" WHERE ({where_sql}) LIMIT {limit_n}'

    def _build_token_search_sql(self, table: str, prompt: str, limit_n: int = 100) -> Optional[str]:
        tokens = [t for t in _tokenize(_normalize_token_variants(prompt.lower())) if t not in STOPWORDS and len(t) > 1]
        text_cols = self._text_columns(table)
        if not tokens or not text_cols:
            return None
        clauses = []
        for tok in tokens[:8]:
            safe_tok = tok.replace("'", "''")
            per_col = [f'LOWER(CAST("{c}" AS VARCHAR)) LIKE LOWER(\'%{safe_tok}%\')' for c in text_cols]
            clauses.append("(" + " OR ".join(per_col) + ")")
        where_sql = " AND ".join(clauses[:3]) if clauses else ""
        if not where_sql:
            return None
        return f'SELECT * FROM "{table}" WHERE {where_sql} LIMIT {limit_n}'

    def _extract_only_column_phrase(self, q: str) -> Optional[str]:
        q = _normalize_token_variants(q)
        m = re.search(r"\bonly\s+([a-zA-Z0-9_ \-]+)", q.lower())
        if m:
            phrase = m.group(1).strip()
            # "only between 2024" is a range constraint, not a projection request.
            if any(k in phrase for k in ["between", "less than", "greater than", "before", "after", "<", ">"]):
                return None
            phrase = re.sub(r"\b(data|rows|record|records)\b", "", phrase).strip()
            phrase = _clean_column_phrase(phrase)
            return phrase if phrase else None
        return None

    def _extract_multi_column_request(self, q: str) -> List[str]:
        qn = _normalize_token_variants(q.lower())
        m = re.search(r"(?:has|show|give)(?:\s+me)?\s+(.+?)\s+(?:column|columns)$", qn)
        if not m:
            m = re.search(r"(?:has|show|give)(?:\s+me)?\s+(.+?)$", qn)
        if not m:
            return []
        phrase = m.group(1)
        if " and " not in phrase:
            return []
        parts = [p.strip() for p in phrase.split(" and ") if p.strip()]
        cleaned = [_clean_column_phrase(p) for p in parts]
        return [c for c in cleaned if c]

    def _date_columns(self, table: str) -> List[str]:
        if table not in self.tables:
            return []
        cols = []
        for c in self.tables[table].columns:
            lc = c.lower()
            if any(k in lc for k in ["date", "time", "year", "admission", "discharge", "dob", "birth", "created", "updated"]):
                cols.append(c)
        return cols

    def _extract_date_condition(self, q: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
        qn = _normalize_token_variants(q.lower())
        # between 2024 and 2025 / between 2024 year
        m = re.search(r"([a-zA-Z0-9_ ]*date[a-zA-Z0-9_ ]*|admission date|discharge date|dob|birth date)?\s*between\s+(\d{4})(?:\s*(?:and|-|to)\s*(\d{4}))?", qn)
        if m:
            col_hint = (m.group(1) or "date").strip()
            y1 = m.group(2)
            y2 = m.group(3) or y1
            return col_hint, "between_years", y1, y2

        m = re.search(r"([a-zA-Z0-9_ ]*date[a-zA-Z0-9_ ]*|admission date|discharge date|dob|birth date)?\s*(?:less than|before|<)\s*(\d{4})", qn)
        if m:
            col_hint = (m.group(1) or "date").strip()
            return col_hint, "lt_year", m.group(2), None

        m = re.search(r"([a-zA-Z0-9_ ]*date[a-zA-Z0-9_ ]*|admission date|discharge date|dob|birth date)?\s*(?:greater than|after|>)\s*(\d{4})", qn)
        if m:
            col_hint = (m.group(1) or "date").strip()
            return col_hint, "gt_year", m.group(2), None
        return None

    def _extract_row_limit(self, q: str) -> Optional[int]:
        qn = _normalize_token_variants(q.lower())
        patterns = [
            r"\btop\s+(\d+)\b",
            r"\bshow\s+me\s+(\d+)\s+(?:rows|records)\b",
            r"\b(\d+)\s+(?:rows|records)\b",
            r"\blimit\s+(\d+)\b",
        ]
        for p in patterns:
            m = re.search(p, qn)
            if m:
                return max(1, min(1000, int(m.group(1))))
        return None
        return None

    def _extract_free_value(self, q: str) -> Optional[str]:
        # Handles prompts like "patients whose blood type is b-" and similar forms.
        m = re.search(r"(?:\bis\b|=)\s*['\"]?([a-zA-Z0-9_ .+\-]+)['\"]?", q.lower())
        if m:
            v = m.group(1).strip()
            return v if len(v) >= 2 else None
        tokens = _tokenize(q)
        if tokens:
            tail = tokens[-1]
            if len(tail) >= 2:
                return tail
        return None

    def _find_table_for_filter(self, raw_col: str, raw_val: str, preferred_table: str) -> Tuple[str, Optional[str]]:
        # 1) Try preferred table directly.
        col = self._best_column_match(preferred_table, raw_col, numeric_only=False)
        if col:
            return preferred_table, col
        # 2) Try value-aware match across all tables.
        best_t, best_c = preferred_table, None
        for t in self._all_table_names():
            c = self._best_column_for_value(t, raw_val)
            if c:
                return t, c
        # 3) Fallback to best column phrase match across tables.
        best_score = -1
        q_tokens = set(_tokenize(raw_col))
        for t in self._all_table_names():
            for c in self.tables[t].columns:
                score = len(q_tokens & set(_tokenize(c)))
                if score > best_score:
                    best_score = score
                    best_t, best_c = t, c
        return best_t, best_c

    def _schema_prompt(self) -> str:
        parts = []
        for tname, df in self.tables.items():
            cols = ", ".join([f'"{c}"' for c in df.columns[:60]])
            parts.append(f'Table "{tname}" columns: {cols}')
        return "\n".join(parts)

    def _validate_sql(self, sql: str) -> str:
        sql_clean = sql.strip().strip("`")
        if not sql_clean.lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")
        # Validate by parsing/executing limited preview in DuckDB.
        self.conn.execute(f"SELECT * FROM ({sql_clean}) AS _x LIMIT 1")
        return sql_clean

    def _llm_generate_sql(self, prompt: str) -> Tuple[str, str]:
        if not self._llm_client:
            raise ValueError("LLM client is not configured.")
        system = (
            "You are an expert text-to-SQL generator for DuckDB.\n"
            "Return ONLY SQL as plain text.\n"
            "Use exact table and column names from schema.\n"
            "For value filters, use case-insensitive matching with LOWER(CAST(col AS VARCHAR)).\n"
            "Never generate non-SELECT statements.\n"
            "Add LIMIT 200 when selecting rows without explicit limit."
        )
        user = (
            f"Schema:\n{self._schema_prompt()}\n\n"
            f"Question: {prompt}\n\n"
            "Generate one DuckDB SELECT query."
        )
        resp = self._llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        sql = (resp.choices[0].message.content or "").strip()
        sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.DOTALL).strip()
        sql = self._validate_sql(sql)
        return sql, "Used LLM-based prompt interpretation and schema-aware SQL planning."

    def generate_sql(self, prompt: str) -> Tuple[str, str, List[str]]:
        q = _normalize_token_variants(prompt.strip().lower())
        warnings = []
        self.last_filter_mode = "exact"
        if self.llm_enabled and self._llm_client is not None:
            try:
                sql, explanation = self._llm_generate_sql(prompt)
                self.last_llm_status = "LLM query planning active."
                return sql, explanation, warnings
            except Exception as ex:
                msg = str(ex)
                # Auto-disable noisy failing LLM mode for this session and continue locally.
                if ("insufficient_quota" in msg) or ("429" in msg) or ("billing" in msg.lower()) or ("invalid_api_key" in msg):
                    self.llm_enabled = False
                    self._llm_client = None
                    self.last_llm_status = "LLM unavailable (quota/auth). Auto-switched to local planner."
                else:
                    self.last_llm_status = "LLM unavailable. Auto-switched to local planner."

        table = self._best_table_match(q)
        if not table:
            return "SELECT 1", "No tables available.", ["No uploaded tables found."]

        intent = "list"
        if any(w in q for w in ["average", "mean", "avg"]):
            intent = "avg"
        elif any(w in q for w in ["sum", "total"]):
            intent = "sum"
        elif any(w in q for w in ["count", "how many", "number of"]):
            intent = "count"
        elif any(w in q for w in ["max", "highest", "largest"]):
            intent = "max"
        elif any(w in q for w in ["min", "lowest", "smallest"]):
            intent = "min"

        if intent in {"avg", "sum", "max", "min"}:
            col = self._best_column_match(table, q, numeric_only=True)
            if not col:
                col = self._best_column_match(table, q, numeric_only=False)
                warnings.append("No clear numeric column found; used best text match.")
            sql = f'SELECT {intent.upper()}("{col}") AS value FROM "{table}"'
            explanation = f"Computed `{intent}` for `{col}` from `{table}`."
        elif intent == "count":
            sql = f'SELECT COUNT(*) AS count FROM "{table}"'
            explanation = f"Counted records from `{table}`."
        else:
            sql = f'SELECT * FROM "{table}"'
            explanation = f"Showing rows from `{table}`."

        # Handle prompts like "give me sales data which has only units sold".
        only_phrase = self._extract_only_column_phrase(q)
        if intent == "list" and only_phrase:
            only_col = self._best_column_match(table, only_phrase, numeric_only=False)
            if only_col:
                sql = f'SELECT "{only_col}" FROM "{table}"'
                explanation = f"Showing only `{only_col}` from `{table}`."
        # Handle prompts like "has Name and Blood Type column".
        multi_cols = self._extract_multi_column_request(q)
        if intent == "list" and multi_cols:
            selected = []
            for phrase in multi_cols:
                c = self._best_column_match(table, phrase, numeric_only=False)
                if c and c not in selected:
                    selected.append(c)
            if selected:
                cols_sql = ", ".join([f'"{c}"' for c in selected])
                sql = f'SELECT {cols_sql} FROM "{table}"'
                explanation = f"Showing selected columns from `{table}`: {', '.join(selected)}."

        # Handle date-specific constraints (between/before/after year).
        date_cond = self._extract_date_condition(q)
        if date_cond:
            col_hint, op, y1, y2 = date_cond
            date_col = self._best_column_match(table, col_hint, numeric_only=False)
            if not date_col:
                dcols = self._date_columns(table)
                date_col = dcols[0] if dcols else None
            if date_col:
                sql = re.sub(r"\s+LIMIT\s+\d+\s*$", "", sql, flags=re.IGNORECASE)
                if op == "between_years":
                    sql += (
                        f' WHERE EXTRACT(YEAR FROM TRY_CAST("{date_col}" AS DATE)) '
                        f"BETWEEN {int(y1)} AND {int(y2)}"
                    )
                    explanation += f" Applied date-year range `{int(y1)} to {int(y2)}` on `{date_col}`."
                elif op == "lt_year":
                    sql += f' WHERE EXTRACT(YEAR FROM TRY_CAST("{date_col}" AS DATE)) < {int(y1)}'
                    explanation += f" Applied date-year filter `< {int(y1)}` on `{date_col}`."
                elif op == "gt_year":
                    sql += f' WHERE EXTRACT(YEAR FROM TRY_CAST("{date_col}" AS DATE)) > {int(y1)}'
                    explanation += f" Applied date-year filter `> {int(y1)}` on `{date_col}`."
                sql += " LIMIT 100"
                return sql, explanation, warnings

        cond = self._extract_condition_value(q)
        if cond:
            raw_col, raw_val = cond
            table, match_col = self._find_table_for_filter(raw_col, raw_val, table)
            if match_col:
                val_clean = _clean_filter_value(raw_col, raw_val)
                val = val_clean.replace("'", "''")
                sql = re.sub(r'FROM\s+"[^"]+"', f'FROM "{table}"', sql, count=1, flags=re.IGNORECASE)
                if " where " in sql.lower():
                    sql += f' AND LOWER(CAST("{match_col}" AS VARCHAR)) = LOWER(\'{val}\')'
                else:
                    sql += f' WHERE LOWER(CAST("{match_col}" AS VARCHAR)) = LOWER(\'{val}\')'
                explanation += f" Applied filter `{match_col} = {val_clean}`."
            else:
                warnings.append("Filter detected but column match was weak; filter skipped.")
        else:
            contains_val = self._extract_contains_value(q)
            if contains_val:
                # Find best table+column from sampled values for contains phrase.
                best_table = table
                best_col = None
                best_score = 0
                needle = contains_val.lower()
                for t in self._all_table_names():
                    for c, vals in self._value_cache.get(t, {}).items():
                        score = 0
                        for v in vals:
                            lv = v.lower()
                            if needle in lv:
                                score += 2
                            elif any(tok in lv for tok in _tokenize(needle)):
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_table = t
                            best_col = c
                if best_col:
                    table = best_table
                    safe_val = contains_val.replace("'", "''")
                    sql = re.sub(r'FROM\s+"[^"]+"', f'FROM "{table}"', sql, count=1, flags=re.IGNORECASE)
                    if " where " in sql.lower():
                        sql += f' AND LOWER(CAST("{best_col}" AS VARCHAR)) LIKE LOWER(\'%{safe_val}%\')'
                    else:
                        sql += f' WHERE LOWER(CAST("{best_col}" AS VARCHAR)) LIKE LOWER(\'%{safe_val}%\')'
                    self.last_filter_mode = "contains"
                    explanation += f" Applied contains filter `{best_col} contains {contains_val}`."
                else:
                    warnings.append("Contains phrase found but no matching column values were detected.")

            maybe_value = self._extract_free_value(q)
            if maybe_value:
                val_col = self._best_column_for_value(table, maybe_value)
                if val_col:
                    val = maybe_value.replace("'", "''")
                    if " where " in sql.lower():
                        sql += f' AND LOWER(CAST("{val_col}" AS VARCHAR)) = LOWER(\'{val}\')'
                    else:
                        sql += f' WHERE LOWER(CAST("{val_col}" AS VARCHAR)) = LOWER(\'{val}\')'
                    explanation += f" Applied value-aware filter `{val_col} = {maybe_value}`."

        row_limit = self._extract_row_limit(q)
        if row_limit is not None:
            sql = re.sub(r"\s+LIMIT\s+\d+\s*$", "", sql, flags=re.IGNORECASE)
            sql += f" LIMIT {row_limit}"
            explanation += f" Limited to first {row_limit} rows."
        elif intent == "list":
            sql += " LIMIT 100"
            explanation += " Limited to first 100 rows for speed."

        return sql, explanation, warnings

    def ask(self, prompt: str) -> QueryResult:
        sql, explanation, warnings = self.generate_sql(prompt)
        try:
            df = self.conn.execute(sql).df()
            if df.empty:
                # Recovery 1: phrase contains-search across all text columns.
                table = self._best_table_match(prompt)
                value_phrase = self._extract_value_phrase(prompt)
                if table and value_phrase:
                    alt_sql = self._build_contains_search_sql(table, value_phrase, limit_n=100)
                    if alt_sql:
                        alt_df = self.conn.execute(alt_sql).df()
                        if not alt_df.empty:
                            warnings.append("No rows from initial SQL; used contains-search recovery.")
                            return QueryResult(
                                sql=alt_sql,
                                explanation="Recovered with semantic contains-search across text columns.",
                                dataframe=alt_df,
                                warnings=warnings,
                            )
                # Recovery 2: token-based broad text search.
                if table:
                    alt_sql2 = self._build_token_search_sql(table, prompt, limit_n=100)
                    if alt_sql2:
                        alt_df2 = self.conn.execute(alt_sql2).df()
                        if not alt_df2.empty:
                            warnings.append("No rows from initial SQL; used token-search recovery.")
                            return QueryResult(
                                sql=alt_sql2,
                                explanation="Recovered with token-based broad search.",
                                dataframe=alt_df2,
                                warnings=warnings,
                            )
            return QueryResult(sql=sql, explanation=explanation, dataframe=df, warnings=warnings)
        except Exception as ex:
            fixed_sql = sql.replace("AVG(", "avg(").replace("SUM(", "sum(").replace("MIN(", "min(").replace("MAX(", "max(")
            try:
                df = self.conn.execute(fixed_sql).df()
                warnings.append(f"Auto-corrected SQL syntax: {ex}")
                return QueryResult(sql=fixed_sql, explanation=explanation, dataframe=df, warnings=warnings)
            except Exception as ex2:
                fallback = f'SELECT * FROM "{self._best_table_match(prompt) or list(self.tables.keys())[0]}" LIMIT 20'
                df = self.conn.execute(fallback).df()
                warnings.append(f"SQL generation failed and fallback was used: {ex2}")
                return QueryResult(sql=fallback, explanation="Fallback preview query executed.", dataframe=df, warnings=warnings)
