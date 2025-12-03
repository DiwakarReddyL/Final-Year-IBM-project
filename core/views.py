import os
import json
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback

from io import BytesIO
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils.safestring import mark_safe
from django.urls import reverse

from .forms import CSVUploadForm
from .models import UploadedCSV
from bert_model.classifier import predict_intent
from core.query_utils import parse_query_to_action

# ---------------- Helper Functions ----------------
def render_plot_base64(fig):
    """
    Convert a matplotlib figure to base64 string for dashboard rendering.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def execute_action(df, action):
    """
    Execute structured action on a Pandas DataFrame.
    Returns a tuple: (rtype, payload)
    rtype: 'html', 'plot', 'scalar', 'text'
    payload: content to render
    """
    try:
        act = action.get('action')

        # ---------------- SIMPLE ACTIONS ----------------
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
            n = action.get('n', 10)
            if col not in df.columns:
                return 'text', f"Column '{col}' not found."
            vals = df[col].dropna().unique().tolist()
            if len(vals) > n:
                return 'html', df[[col]].head(n).to_html(classes="table table-bordered", index=False)
            else:
                return 'text', ", ".join(map(str, vals[:n]))

        # ---------------- AGGREGATIONS ----------------
        if act == 'aggregate':
            col = action.get('col')
            agg = action.get('agg')
            if not col or col not in df.columns:
                numeric = df.select_dtypes(include=['number']).columns.tolist()
                if not numeric:
                    return 'text', "No numeric columns to aggregate."
                col = numeric[0]
            s = df[col]
            if agg == 'mean': return 'scalar', float(s.mean())
            if agg == 'sum': return 'scalar', float(s.sum())
            if agg == 'max': return 'scalar', float(s.max())
            if agg == 'min': return 'scalar', float(s.min())
            if agg == 'count': return 'scalar', int(s.count())
            return 'html', s.describe().to_frame().to_html(classes="table table-bordered")

        # ---------------- FILTERS ----------------
        if act == 'filter_numeric':
            col, op, val = action.get('col'), action.get('op'), action.get('value')
            if col not in df.columns:
                return 'text', f"Column '{col}' not found."
            ser = pd.to_numeric(df[col], errors='coerce')
            if op == '<': res = df[ser < val]
            elif op == '<=': res = df[ser <= val]
            elif op == '>': res = df[ser > val]
            elif op == '>=': res = df[ser >= val]
            elif op == '!=': res = df[ser != val]
            else: res = df[ser == val]
            if res.empty:
                return 'text', "No matching rows found."
            return 'html', res.head(200).to_html(classes="table table-bordered", index=False)

        if act == 'filter_equals':
            col, val = action.get('col'), action.get('value')
            if col not in df.columns:
                return 'text', f"Column '{col}' not found."
            res = df[df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()]
            if res.empty:
                return 'text', "No matching rows found."
            return 'html', res.head(200).to_html(classes="table table-bordered", index=False)

        if act == 'filter_multi_equals':
            filtered_df = df.copy()
            for col, val in action.get('conditions', []):
                col_match = next((c for c in filtered_df.columns if c.lower() == col.lower()), None)
                if col_match:
                    filtered_df = filtered_df[
                        filtered_df[col_match].astype(str).str.strip().str.lower() == str(val).strip().lower()
                    ]
            if filtered_df.empty:
                return 'text', "No matching rows found."
            return 'html', filtered_df.head(200).to_html(classes="table table-bordered", index=False)

        # ---------------- SORT / TOP ----------------
        if act == 'sort':
            col = action.get('col')
            asc = action.get('ascending', True)
            if col not in df.columns:
                return 'text', f"Column '{col}' not found."
            return 'html', df.sort_values(col, ascending=asc).head(200).to_html(classes="table table-bordered", index=False)

        if act == 'top':
            col = action.get('col')
            n = action.get('n', 1)
            asc = action.get('ascending', False)
            if col not in df.columns:
                numerics = df.select_dtypes(include=['number']).columns.tolist()
                if not numerics:
                    return 'text', "No numeric columns found to compute top."
                col = numerics[0]
            return 'html', df.sort_values(col, ascending=asc).head(n).to_html(classes="table table-bordered", index=False)

        if act == 'group_top':
            group_col = action.get('group_col')
            col = action.get('col')
            n = action.get('n', 1)
            asc = action.get('ascending', False)
            if group_col not in df.columns:
                return 'text', f"Group column '{group_col}' not found."
            if not col or col not in df.columns:
                numerics = df.select_dtypes(include=['number']).columns.tolist()
                if not numerics:
                    return 'text', "No numeric columns to aggregate in group."
                col = numerics[0]
            grouped = df.sort_values(col, ascending=asc).groupby(group_col, as_index=False).head(n)
            if grouped.empty:
                return 'text', "No matching rows found in group."
            return 'html', grouped.to_html(classes="table table-bordered", index=False)

        # ---------------- COUNT CONDITION ----------------
        if act == 'count_condition':
            col, op, val = action.get('col'), action.get('op'), action.get('value')
            if col not in df.columns:
                return 'text', f"Column '{col}' not found."
            s = pd.to_numeric(df[col], errors='coerce')
            if op == '<': cnt = int((s < val).sum())
            elif op == '<=': cnt = int((s <= val).sum())
            elif op == '>': cnt = int((s > val).sum())
            elif op == '>=': cnt = int((s >= val).sum())
            elif op == '!=': cnt = int((s != val).sum())
            else: cnt = int((s == val).sum())
            return 'scalar', cnt

        # ---------------- PLOTS ----------------
        if act == 'plot':
            kind = action.get('kind', 'bar')
            x = action.get('x')
            y = action.get('y')

            if y not in df.columns:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                y = numeric_cols[0] if numeric_cols else None

            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                if kind == 'bar':
                    if x and y and x in df.columns and y in df.columns:
                        grouped = df.groupby(x)[y].mean().sort_values(ascending=False).head(20)
                        grouped.plot(kind='bar', ax=ax)
                        ax.set_ylabel(y)
                        ax.set_xlabel(x)
                    elif x and x in df.columns:
                        df[x].value_counts().head(20).plot(kind='bar', ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel(x)
                    elif y and y in df.columns:
                        df[y].dropna().head(50).plot(kind='bar', ax=ax)
                    else:
                        return 'text', "No valid data found to plot a bar chart."

                elif kind == 'line':
                    df[y].dropna().head(200).plot(kind='line', ax=ax)

                elif kind == 'hist':
                    df[y].dropna().plot(kind='hist', bins=20, ax=ax)

                elif kind == 'scatter':
                    if x and y and x in df.columns and y in df.columns:
                        df.plot(kind='scatter', x=x, y=y, ax=ax)
                    else:
                        return 'text', "Scatter plot requires both x and y columns."

                ax.set_title(f"{kind.title()} Plot ({x or y})")
                img = render_plot_base64(fig)
                return 'plot', img

            except Exception as e:
                plt.close(fig)
                return 'text', f"Plot generation failed: {str(e)}"

        return 'text', "I couldn't map that request to an operation. Try phrasing differently."

    except Exception as e:
        tb = traceback.format_exc()
        return 'text', f"Execution error: {str(e)}\n{tb}"


# ---------------- DASHBOARD VIEW ----------------
@login_required
def dashboard(request):
    user = request.user
    form = CSVUploadForm()
    df = None
    df_json = []
    error = None
    query_error = None
    preview_id = request.GET.get("preview_id")
    query_result = None
    query_image = None
    query_text = None
    query_code = ""

    if request.method == 'POST':
        # -------- CSV UPLOAD --------
        if 'file' in request.FILES:
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                instance = form.save(commit=False)
                instance.user = user
                instance.save()
                return redirect(reverse('core:dashboard') + f"?preview_id={instance.id}")

        # -------- QUERY EXECUTION --------
        elif 'query_text' in request.POST:
            query = request.POST.get('query_text', '').strip()
            preview_id = request.POST.get('preview_id')
            try:
                if not preview_id:
                    raise ValueError("No CSV selected. Upload or select a CSV first.")

                csv_obj = UploadedCSV.objects.get(id=preview_id, user=user)
                df = pd.read_csv(csv_obj.file.path)

                intent = predict_intent(query)
                action = parse_query_to_action(intent, query, df)
                query_code = json.dumps(action, indent=2)

                rtype, payload = execute_action(df, action)
                if rtype == 'html':
                    query_result = payload
                elif rtype == 'plot':
                    query_image = payload
                elif rtype in ('scalar', 'text'):
                    query_text = str(payload)
                else:
                    query_text = str(payload)

            except Exception as e:
                query_error = f"Query failed: {str(e)}"

    # -------- CSV PREVIEW --------
    elif preview_id:
        try:
            csv_obj = UploadedCSV.objects.get(id=preview_id, user=user)
            df = pd.read_csv(csv_obj.file.path)
            df_json = df.head(5).values.tolist()
        except Exception as e:
            error = f"Preview error: {e}"

    history = UploadedCSV.objects.filter(user=user).order_by('-timestamp')

    return render(request, 'core/dashboard.html', {
        'form': form,
        'df': df,
        'df_json': mark_safe(json.dumps(df_json)),
        'error': error,
        'history': history,
        'user_info': user.email,
        'preview_id': preview_id,
        'query_code': query_code,
        'query_error': query_error,
        'query_result': query_result,
        'query_text': query_text,
        'query_image': query_image,
    })


# ---------------- DELETE CSV ----------------
@login_required
def delete_csv(request, file_id):
    file = get_object_or_404(UploadedCSV, id=file_id, user=request.user)
    file.file.delete(save=False)
    file.delete()
    return redirect('core:dashboard')
