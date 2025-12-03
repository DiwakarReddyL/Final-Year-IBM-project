import re
from fuzzywuzzy import process
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import base64
from io import BytesIO

# ---------------- CONFIG ----------------
FUZZY_THRESHOLD = 60
TOP_N_DEFAULT = 1
MAX_DISPLAY_ROWS = 200  # Max rows to show in table
MAX_PLOT_CATEGORIES = 20  # Limit for bar/pie plots

# ---------------- HELPERS ----------------
def _normalize(text):
    return text.lower().strip()

def fuzzy_match_column(query_fragment, columns):
    """Fuzzy match query fragment to column names"""
    if not columns:
        return None, 0
    query_fragment_norm = _normalize(query_fragment)
    columns_norm = [c.lower() for c in columns]
    result = process.extractOne(query_fragment_norm, columns_norm)
    if not result:
        return None, 0
    col_norm, score = result
    if score >= FUZZY_THRESHOLD:
        return columns[columns_norm.index(col_norm)], score
    return None, score

def find_mentioned_columns(query, columns, max_matches=3):
    """Extract mentioned columns from query using fuzzy matching"""
    q = _normalize(query)
    found = []

    # Quoted columns
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    for pair in quoted:
        colname = pair[0] or pair[1]
        if colname:
            col, _ = fuzzy_match_column(colname, columns)
            if col and col not in found:
                found.append(col)

    # Token matching (3-grams, 2-grams, 1-grams)
    tokens = re.findall(r"[A-Za-z0-9_]+", q)
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            fragment = " ".join(tokens[i:i+n])
            col, _ = fuzzy_match_column(fragment, columns)
            if col and col not in found:
                found.append(col)
                if len(found) >= max_matches:
                    return found

    # fallback full query match
    col, _ = fuzzy_match_column(q, columns)
    if col and col not in found:
        found.append(col)
    return found

def extract_number_and_comparator(query):
    """Extract number and comparator from query"""
    q = query.lower()
    comp = None
    if re.search(r"(less than or equal|<=|at most)", q):
        comp = "<="
    elif re.search(r"(less than|below|under|<)", q):
        comp = "<"
    elif re.search(r"(greater than or equal|>=|at least)", q):
        comp = ">="
    elif re.search(r"(greater than|above|over|>)", q):
        comp = ">"
    elif re.search(r"(not equal|!=|not equal to)", q):
        comp = "!="
    elif re.search(r"(equal to|equals|=|==)", q):
        comp = "=="

    m = re.search(r"(-?\d+(\.\d+)?%?)", q)
    val = None
    if m:
        raw = m.group(1)
        if raw.endswith('%'):
            try:
                val = float(raw[:-1]) / 100.0
            except:
                val = None
        else:
            try:
                val = float(raw)
            except:
                val = None
    return val, comp

def extract_top_n(query):
    """Extract top N if mentioned in query"""
    m = re.search(r"top\s+(\d+)", query.lower())
    if m:
        return int(m.group(1))
    m2 = re.search(r"first\s+(\d+)", query.lower())
    if m2:
        return int(m2.group(1))
    if re.search(r"\b(top|highest|maximum|max|best|largest)\b", query.lower()):
        return TOP_N_DEFAULT
    return None

def detect_plot_type(query):
    """Detect plot type from query"""
    q = query.lower()
    if "pie" in q:
        return "pie"
    if "bar" in q:
        return "bar"
    if "line" in q:
        return "line"
    if "hist" in q or "histogram" in q:
        return "hist"
    if "scatter" in q:
        return "scatter"
    if "plot" in q or "chart" in q or "visual" in q or "graph" in q:
        return "bar"
    return None

def detect_aggregation(query):
    """Detect aggregation function"""
    q = query.lower()
    if re.search(r"\b(mean|average)\b", q): return "mean"
    if re.search(r"\b(sum|total)\b", q): return "sum"
    if re.search(r"\b(max|maximum|highest)\b", q): return "max"
    if re.search(r"\b(min|minimum|lowest)\b", q): return "min"
    if re.search(r"\b(count|how many|number of)\b", q): return "count"
    return None

def extract_multiple_conditions(query, columns):
    """Extract multiple string conditions from query"""
    q = query.lower()
    parts = re.split(r"\band\b|\bor\b|,|&", q)
    conditions = []
    for part in parts:
        # string equality conditions
        m = re.search(r"([a-zA-Z0-9 _]+)\s+(is|equals|=)\s+([a-zA-Z0-9 _.-]+)", part.strip())
        if m:
            col_fragment = m.group(1).strip()
            val = m.group(3).strip()
            col, _ = fuzzy_match_column(col_fragment, columns)
            if col:
                conditions.append((col, val))
    return conditions

def render_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ---------------- PARSER ----------------
def parse_query_to_action(intent, query, df):
    """Convert query + intent to structured action"""
    columns = df.columns.tolist()
    q = _normalize(query)
    mentioned = find_mentioned_columns(query, columns, max_matches=3)
    conditions = extract_multiple_conditions(query, columns)

    # basic actions
    if intent in ("columns", "list_columns"):
        return {'action':'columns'}
    if intent in ("count", "num_rows"):
        return {'action':'count'}
    if intent in ("describe",):
        return {'action':'describe'}

    # aggregation
    agg = detect_aggregation(query)
    if agg:
        col = mentioned[0] if mentioned else None
        return {'action':'aggregate','agg':agg,'col':col}

    # plotting
    plot_kind = detect_plot_type(query)
    if plot_kind:
        x, y = (mentioned[0], mentioned[1]) if len(mentioned) >= 2 else (mentioned[0] if mentioned else None, None)
        return {'action':'plot','kind':plot_kind,'x':x,'y':y,'conditions':conditions}

    # filtering
    if len(conditions) > 1:
        return {'action':'filter_multi_equals','conditions':conditions}
    if len(conditions) == 1:
        col, val = conditions[0]
        return {'action':'filter_equals','col':col,'value':val}

    # numeric filters
    val, comp = extract_number_and_comparator(query)
    if val is not None and comp is not None and mentioned:
        return {'action':'filter_numeric','col':mentioned[0],'op':comp,'value':val}

    # sorting
    if "sort" in q or "order by" in q:
        if mentioned:
            ascending = not any(w in q for w in ("desc","descending","high to low"))
            return {'action':'sort','col':mentioned[0],'ascending':ascending}

    # fallback
    if any(w in q for w in ("show","list","display","give")):
        if mentioned:
            return {'action':'select_columns','columns':mentioned}
        return {'action':'head','n':5}

    if mentioned:
        return {'action':'select_column_values','col':mentioned[0],'n':10}

    return {'action':'head','n':5}

# ---------------- EXECUTOR ----------------
def execute_action(df, action):
    """Execute structured action on dataframe"""
    try:
        act = action.get('action')

        if act == 'columns':
            return 'text', ", ".join(df.columns.tolist())
        if act == 'count':
            return 'scalar', len(df)
        if act == 'describe':
            return 'html', df.describe(include='all').to_html(classes="table table-bordered")
        if act == 'head':
            n = action.get('n', 5)
            return 'html', df.head(n).to_html(classes="table table-bordered", index=False)
        if act == 'select_columns':
            cols = [c for c in action.get('columns', []) if c in df.columns]
            if not cols:
                return 'text', "No matching columns found."
            return 'html', df[cols].head(50).to_html(classes="table table-bordered", index=False)
        if act == 'select_column_values':
            col = action.get('col')
            if col not in df.columns:
                return 'text', f"Column '{col}' not found."
            vals = df[col].dropna().unique().tolist()
            return 'text', ", ".join(map(str, vals[:10]))

        # Aggregations
        if act == 'aggregate':
            col = action.get('col')
            agg = action.get('agg')
            if not col or col not in df.columns:
                return 'text', "Aggregation requires a valid column."
            s = df[col]
            try:
                if agg == 'mean': return 'scalar', float(s.mean())
                if agg == 'sum': return 'scalar', float(s.sum())
                if agg == 'max': return 'scalar', float(s.max())
                if agg == 'min': return 'scalar', float(s.min())
                if agg == 'count': return 'scalar', int(s.count())
            except:
                return 'text', f"Cannot perform '{agg}' on column '{col}'"
            return 'text', f"Aggregation '{agg}' not supported."

        # Filtering
        if act == 'filter_equals':
            col, val = action.get('col'), action.get('value')
            res = df[df[col].astype(str).str.lower() == str(val).lower()]
            return 'html', res.head(MAX_DISPLAY_ROWS).to_html(classes="table table-bordered", index=False)
        if act == 'filter_multi_equals':
            res = df.copy()
            for col, val in action.get('conditions', []):
                if col in res.columns:
                    res = res[res[col].astype(str).str.lower() == str(val).lower()]
            return 'html', res.head(MAX_DISPLAY_ROWS).to_html(classes="table table-bordered", index=False)
        if act == 'filter_numeric':
            col, op, val = action['col'], action['op'], action['value']
            if op == '<': res = df[df[col] < val]
            elif op == '<=': res = df[df[col] <= val]
            elif op == '>': res = df[df[col] > val]
            elif op == '>=': res = df[df[col] >= val]
            elif op == '!=': res = df[df[col] != val]
            else: res = df[df[col] == val]
            return 'html', res.head(MAX_DISPLAY_ROWS).to_html(classes="table table-bordered", index=False)

        # Sorting
        if act == 'sort':
            col = action['col']
            asc = action.get('ascending', True)
            return 'html', df.sort_values(col, ascending=asc).head(MAX_DISPLAY_ROWS).to_html(classes="table table-bordered", index=False)

        # Count condition
        if act == 'count_condition':
            col, op, val = action['col'], action['op'], action['value']
            s = df[col]
            if op == '<': cnt = int((s < val).sum())
            elif op == '<=': cnt = int((s <= val).sum())
            elif op == '>': cnt = int((s > val).sum())
            elif op == '>=': cnt = int((s >= val).sum())
            elif op == '!=': cnt = int((s != val).sum())
            else: cnt = int((s == val).sum())
            return 'scalar', cnt

        # Plots
        if act == 'plot':
            kind = action.get('kind', 'bar')
            x = action.get('x')
            y = action.get('y')
            conditions = action.get('conditions', [])
            fig, ax = plt.subplots(figsize=(6,4))
            df_plot = df.copy()

            # Apply filters
            for col_cond, val_cond in conditions:
                if col_cond in df_plot.columns:
                    df_plot = df_plot[df_plot[col_cond].astype(str).str.lower() == str(val_cond).lower()]

            # Pie chart
            if kind == 'pie':
                if not x or x not in df_plot.columns:
                    return 'text', "Pie chart requires a categorical column."
                counts = df_plot[x].value_counts().head(MAX_PLOT_CATEGORIES)
                counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel('')
                ax.set_title(f"Pie Chart of {x}")
                img = render_plot_base64(fig)
                return 'plot', img

            # Bar chart
            elif kind == 'bar':
                if x and not y:
                    counts = df_plot[x].value_counts().head(MAX_PLOT_CATEGORIES)
                    counts.plot(kind='bar', ax=ax)
                    ax.set_ylabel("Count")
                    ax.set_title(f"Count of {x}")
                    img = render_plot_base64(fig)
                    return 'plot', img
                elif x and y:
                    if pd.api.types.is_numeric_dtype(df_plot[y]):
                        grouped = df_plot.groupby(x)[y].mean().sort_values(ascending=False).head(MAX_PLOT_CATEGORIES)
                        grouped.plot(kind='bar', ax=ax)
                        ax.set_ylabel(f"Average {y}")
                        ax.set_title(f"Average {y} by {x}")
                        img = render_plot_base64(fig)
                        return 'plot', img
                    else:
                        combo = pd.crosstab(df_plot[x], df_plot[y]).head(MAX_PLOT_CATEGORIES)
                        combo.plot(kind='bar', ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_title(f"Counts of {y} by {x}")
                        img = render_plot_base64(fig)
                        return 'plot', img
                else:
                    return 'text', "Bar chart requires at least one column."

            # Line plot
            elif kind == 'line':
                if not y or y not in df_plot.columns:
                    numeric_cols = df_plot.select_dtypes(include='number').columns
                    y = numeric_cols[0] if numeric_cols.any() else None
                if not y:
                    return 'text', "Line plot requires a numeric column."
                df_plot[y].dropna().head(100).plot(ax=ax)
                ax.set_title(f"Line plot of {y}")
                img = render_plot_base64(fig)
                return 'plot', img

            # Histogram
            elif kind == 'hist':
                if not y or y not in df_plot.columns:
                    numeric_cols = df_plot.select_dtypes(include='number').columns
                    y = numeric_cols[0] if numeric_cols.any() else None
                if not y:
                    return 'text', "Histogram requires numeric column."
                df_plot[y].dropna().plot(kind='hist', bins=20, ax=ax)
                ax.set_title(f"Histogram of {y}")
                img = render_plot_base64(fig)
                return 'plot', img

            # Scatter plot
            elif kind == 'scatter':
                if not x or not y:
                    return 'text', "Scatter plot requires two numeric columns."
                if x not in df_plot.columns or y not in df_plot.columns:
                    return 'text', f"Columns '{x}' or '{y}' not found."
                if not pd.api.types.is_numeric_dtype(df_plot[x]) or not pd.api.types.is_numeric_dtype(df_plot[y]):
                    return 'text', "Scatter plot requires numeric columns for both x and y."
                df_plot.plot(kind='scatter', x=x, y=y, ax=ax)
                ax.set_title(f"Scatter plot ({x} vs {y})")
                img = render_plot_base64(fig)
                return 'plot', img

            else:
                return 'text', f"Plot type '{kind}' not supported."

        return 'text', "Couldn't understand your query."

    except Exception as e:
        tb = traceback.format_exc()
        return 'text', f"Execution error: {str(e)}\n{tb}"
