
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from flask import Flask, request, render_template_string
from src.predict import EssayScorer

app = Flask(__name__)
scorer = EssayScorer()

HTML = '''
<!doctype html>
<title>Essay Grader</title>
<h1>Enter Essay</h1>
<form method=post>
  <textarea name=essay rows=10 cols=50></textarea><br>
  <input type=submit value=Grade>
</form>
{% if score is not none %}
  <h2>Predicted Score: {{ score }}</h2>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    if request.method == 'POST':
        essay = request.form['essay']
        score = scorer.predict(essay)
    return render_template_string(HTML, score=score)

if __name__ == '__main__':
    app.run(debug=True)
