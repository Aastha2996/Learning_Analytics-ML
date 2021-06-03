from flask import Flask

app = Flask(__name__)
@app.before_request
def before():
    print("This is executed BEFORE each request.")


@app.route('/')
def hello():
    return "Hello " 

app.run()    