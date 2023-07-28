from flask import Flask, render_template, jsonify, request
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.logger import logging

application = Flask(__name__)
app = application

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predictData():
  if request.method == "GET":
    return render_template("form.html")
  else:
    try:
      carat = float(request.form["carat"])
      depth = float(request.form["depth"])
      table = float(request.form["table"])
      x = float(request.form["x"])
      y = float(request.form["y"])
      z = float(request.form["z"])
      cut = request.form["cut"]
      color = request.form["color"]
      clarity = request.form["clarity"]
      customData = CustomData(carat, depth, table, x, y, z, cut, color, clarity)
      predPipeline = PredictPipeline()
      pred = predPipeline.predict(customData.getDFForData())
      return render_template("form.html", final_result="Predicted Price is {}".format(pred[0]))
    except Exception as e:
      logging.info("Exception occured in prediction")
      return render_template("index.html", prediction_text="Error occured during prediction {}".format(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5005)
