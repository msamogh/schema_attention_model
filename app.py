from sanic import Sanic
from sanic.response import json

from data_model_utils import load_saved_model
from chat import chat, load_schema_json

DOMAIN = "ride"
TASK = "ride_book"

app = Sanic(__name__)
app.schema = load_schema_json(TASK)
app.model = load_saved_model(task=TASK)


@app.route("/chat", methods=["POST"])
async def chat(request):
    history = request.json["context"]
    user_message = request.json["message"]
    response = await chat.handle_web_message(
        history, user_message, app.model, app.schema, TASK, DOMAIN
    )
    return json(response)


@app.route("/restart_from", methods=["POST"])
async def restart_from(request):
    pass


app.run(host="0.0.0.0", port=5000)
