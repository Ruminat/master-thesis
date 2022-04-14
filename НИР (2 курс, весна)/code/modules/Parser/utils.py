import MeCab

wakati = MeCab.Tagger("-Owakati")
tagger = MeCab.Tagger()

# Text to MeCab tokens:
# "お前" -> [{ "token': "お前", "partOfSpeech": "代名詞" }]
def getMeCabTokens(text: str) -> list[dict]:
  result = []
  for tokenParts in tagger.parse(text).split("\n"):
    parts = tokenParts.split("\t")
    token = parts[0]

    if token == "" or token == "EOS":
      continue

    resultToken = {}
    resultToken["token"] = token

    if len(parts) >= 5:
      resultToken["partOfSpeech"] = parts[4]

    result.append(resultToken)
  return result
