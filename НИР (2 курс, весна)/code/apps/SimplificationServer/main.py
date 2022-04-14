from flask import Flask, request, jsonify
from flask_cors import cross_origin

from modules.Parser.utils import getMeCabTokens
from utils import initiatePyTorch, loadTransformer

def startSimplificationServerApp():
  app = Flask(__name__)

  initiatePyTorch()
  transformer = loadTransformer()

  @app.route("/processJapaneseText", methods=["GET"])
  @cross_origin()
  def getProcessJapaneseText():
    text = request.args.get("text").strip()

    if (text == ""):
      return jsonify({
        "originalText": "",
        "simplifiedText": "",
        "originalTextTokens": [],
        "simplifiedTextTokens": []
      })

    simplifiedText = transformer.translate(text)
    textTokens = getMeCabTokens(text)
    simplifiedTextTokens = getMeCabTokens(simplifiedText)

    return jsonify({
      "originalText": text,
      "simplifiedText": simplifiedText,
      "originalTextTokens": textTokens,
      "simplifiedTextTokens": simplifiedTextTokens
    })

  app.run()
