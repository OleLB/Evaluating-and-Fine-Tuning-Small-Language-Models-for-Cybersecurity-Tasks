import os
import sqlite3
from flask import Flask, render_template, request

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "../db/scores.db")

ALLOWED_SORT_COLUMNS = {"id", "model_name", "LLM_score"}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    model_filter = request.args.get("model", "")
    sort_col = request.args.get("sort", "id")
    order = request.args.get("order", "asc")

    if sort_col not in ALLOWED_SORT_COLUMNS:
        sort_col = "id"
    if order not in ("asc", "desc"):
        order = "asc"

    with get_db() as conn:
        models = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT model_name FROM Scores ORDER BY model_name"
            ).fetchall()
        ]

        query = """
            SELECT Scores.id, Scores.model_name, Inputs.input,
                   Scores.output, Scores.rag_output, Scores.cve_data,
                   Scores.LLM_score, Scores.reasoning
            FROM Scores
            JOIN Inputs ON Scores.input_id = Inputs.id
        """
        params = []
        if model_filter:
            query += " WHERE Scores.model_name = ?"
            params.append(model_filter)

        query += f" ORDER BY Scores.{sort_col} {order.upper()}"

        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    return render_template(
        "index.html",
        rows=rows,
        models=models,
        model_filter=model_filter,
        sort_col=sort_col,
        order=order,
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
