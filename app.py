from core import main
from flask import Flask
app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    result = main().to_html()
    return result


@app.route("/about")
def about():
    return "<h1>About Page</h1>"


if __name__ == '__main__':
    app.run(debug=True)

