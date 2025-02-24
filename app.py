from datetime import datetime
import os, time, json
from flask import Flask, jsonify, request
from supabase import create_client, Client
from flask_cors import CORS
import torch
from model import encode_data, GATMinGRU, decode_event
from langchain_groq import ChatGroq
import dotenv

app = Flask(__name__)
dotenv.load_dotenv()
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["http://localhost:3000"],  # Specific origin instead of *
            # "origins": "*",  # Allow all origins
            "methods": ["GET", "POST", "OPTIONS"],  # Allowed methods
            "allow_headers": ["Content-Type", "Authorization"],  # Common headers
            "supports_credentials": True,
        }
    },
)

# llm = ChatGroq(model="mixtral-8x7b-32768")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# model = GATMinGRU(input_size=166, hidden_size=256, event_embedding_size=16, gat_heads=2)
# model.load_state_dict(torch.load("/Users/fahmiomer/CXC2025/checkpoint_epoch_6.pth")) 
# model.eval() 

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE_NAME = "v2_federato_amplitude_data"
# TABLE_NAME = "v2024_1_federato_amplitude_data"

model = GATMinGRU(input_size=166, hidden_size=512, event_embedding_size=16, gat_heads=4)
checkpoint = torch.load("./model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
h_prev1=torch.zeros(1, 512)
h_prev2=torch.zeros(1, 512)


@app.route("/", methods=["GET"])
def hello():
    return "API is running!"


# returns a json of data from a specifc user's session
@app.route("/model/search/user/<int:user_id>", methods=["GET"])
def get_user_sessions(user_id):
    categories = [
        "amplitude_id",
        "app",
        "region",
        "country",
        "language",
        "device_family",
        "device_type",
        "os_name",
        "session_id",
        "event_type",
        "event_time",
        "platform",
    ]

    dicts = [
        "event_type",
        "region",
        "country",
        "language",
        "device_family",
        "device_type",
        "os_name",
    ]

    user_response = (
        supabase.table("user_table")
        .select(
            "amplitude_id, average_session_time, total_session_time, user_retention_30"
        )
        .eq("user_id", user_id)
        .execute()
    )
    user_response = user_response.data[0]
    amplitude_id = user_response["amplitude_id"]

    dict_mapping = {}
    for category in dicts:
        dict_response = (
            supabase.table(category).select(f"{category}, dict_{category}").execute()
        )
        mapping = {}
        for row in dict_response.data:
            key = row[category]
            value = row[f"dict_{category}"]
            mapping[key] = value
        dict_mapping[category] = mapping

    batch_size = 10000
    offset = 0
    events_response = []
    while True:
        try:
            events_batch = (
                supabase.table(TABLE_NAME)
                .select("*")
                .order("event_time", desc=False)
                .eq("amplitude_id", amplitude_id)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            if not events_batch.data:
                break
            events_response.extend(events_batch.data)
            offset += batch_size
            time.sleep(0.8)
        except Exception as batch_error:
            print(batch_error, "continuing ...")
            time.sleep(5)
    print(f"processed between {len(events_response)} rows")

    events = []
    for event in events_response:
        result = {}
        for category in categories:
            result[category] = event[category]

        for category in dicts:
            result[f"dict_{category}"] = dict_mapping.get(category).get(
                result[category]
            )

        result["average_session_time"] = user_response["average_session_time"]
        result["total_session_time"] = user_response["total_session_time"]
        result["user_retention_30"] = user_response["user_retention_30"]

        events.append(result)

    for event in events:
        if "." in event["event_time"]:
            date_part, ms_part = event["event_time"].split(".")
            ms_part = (ms_part + "000000")[:6]
            event["event_time"] = datetime.fromisoformat(f"{date_part}.{ms_part}")
        else:
            event["event_time"] = datetime.fromisoformat(event["event_time"])

    for i, event in enumerate(events):
        if i > 0:
            event["time_since_last"] = (
                event["event_time"] - events[i - 1]["event_time"]
            ).total_seconds()
            if event["time_since_last"] < 0:
                event["time_since_last"] = None
        else:
            event["time_since_last"] = None

        if i < len(events) - 1:
            event["time_to_next_event"] = (
                events[i + 1]["event_time"] - event["event_time"]
            ).total_seconds()
            if event["time_to_next_event"] < 0:
                event["time_to_next_event"] = None
        else:
            event["time_to_next_event"] = None

        for j in range(1, 5):
            if i - j >= 0:
                event[f"prev_{j}_event_type"] = events[i - j]["event_type"]
                event[f"dict_pet{j}"] = events[i - j]["dict_event_type"]
                event[f"time_since_last_{j}"] = (
                    event["event_time"] - events[i - j]["event_time"]
                ).total_seconds()
                if event[f"time_since_last_{j}"] < 0:
                    event[f"time_since_last_{j}"] = None
            else:
                event[f"prev_{j}_event_type"] = ""
                event[f"dict_pet{j}"] = None
                event[f"time_since_last_{j}"] = None

        if i < len(events) - 1:
            event["next_et"] = events[i + 1]["event_type"]
            event["dict_next_et"] = events[i + 1]["dict_event_type"]
        else:
            event["next_et"] = ""
            event["dict_next_et"] = None

    for event in events:
        event["event_time"] = event["event_time"].isoformat()

    return jsonify(events)


@app.route("/model/search/users/<int:start_id>/<int:end_id>", methods=["GET"])
def get_users_sessions_range(start_id, end_id):
    if start_id > end_id:
        return jsonify({"error": "start_id must be less than or equal to end_id"}), 400

    results = [
        get_user_sessions(user_id).json for user_id in range(start_id, end_id + 1)
    ]

    return jsonify(results)


profile = {}


@app.route("/set_profile", methods=["POST"])
def set_profile():
    global profile
    app_type = request.form.get("app")
    region = request.form.get("region")
    country = request.form.get("country")
    device_family = request.form.get("device_family")
    device_type = request.form.get("device_type")
    os_name = request.form.get("os_name")
    platform = request.form.get("platform")

    if (
        not app_type
        or not region
        or not country
        or not device_family
        or not device_type
        or not os_name
        or not platform
    ):
        return "missing fields", 400

    profile = {
        "app_type": app_type,
        "region": region,
        "country": country,
        "device_family": device_family,
        "device_type": device_type,
        "os_name": os_name,
        "platform": platform,
    }

    return "updated", 201


@app.route("/get_profile", methods=["GET"])
def get_profile():
    return jsonify(profile)


@app.route("/create-session", methods=["GET", "POST"])
def create_user_session():
    dicts = [
        "event_type",
        "region",
        "country",
        "language",
        "device_family",
        "device_type",
        "os_name",
    ]
    data = request.get_json()
    user_id = data["user_id"]
    events = data["events"]
    user = None

    user_response = get_user_chunk(user_id, 0)
    if user_response.data:
        user = json.loads(user_response.data.decode('utf-8'))[0]

    dict_mapping = {}
    for category in dicts:
        dict_response = (
            supabase.table(category).select(f"{category}, dict_{category}").execute()
        )
        mapping = {}
        for row in dict_response.data:
            key = row[category]
            value = row[f"dict_{category}"]
            mapping[key] = value
        dict_mapping[category] = mapping

    event_json = {}

    event_json["dict_next_et"] = None
    event_json["next_et"] = None
    event_json["time_since_last"] = 0
    event_json["time_to_next_event"] = 0
    for j in range(1, 5):
        event_json[f"dict_pet{j}"] = None
        event_json[f"prev_{j}_event_type"] = None
        event_json[f"time_since_last_{j}"] = 0

    event_json["app"] = user["app"]
    event_json["amplitude_id"] = user["amplitude_id"]
    event_json["average_session_time"] = user["average_session_time"]
    event_json["country"] = user["country"]
    event_json["device_family"] = user["device_family"]
    event_json["device_type"] = user["device_type"]
    event_json["language"] = user["language"]
    event_json["event_time"] = user["event_time"]
    event_json["os_name"] = user["os_name"]
    event_json["region"] = user["region"]
    event_json["platform"] = user["platform"]
    event_json["event_type"] = events[4]
    for category in dicts:
        event_json[f"dict_{category}"] = dict_mapping.get(category).get(
            event_json[category]
        )
    
    for i in range(1, 5):
        event_json[f"prev_{i}_event_type"] = events[4 - i]
        event_json[f"dict_pet{i}"] = dict_mapping.get("event_type").get(events[4 - i])

    return jsonify(event_json)

@app.route("/user-data", methods=["POST"])
def get_user_data():
    data = request.get_json()
    user_id = data["user_id"]

    user = supabase.table("user_table").select("*").eq("user_id", user_id).execute()
    if user.data:
        user = user.data[0]
        return jsonify(user)
    return "no user", 400

@app.route("/model/search/user_chunk/<int:refined_user_id>/<int:chunk>", methods=["GET"])
def get_user_chunk(refined_user_id, chunk):
    categories = [
        "amplitude_id",
        "app",
        "region",
        "country",
        "language",
        "device_family",
        "device_type",
        "os_name",
        "session_id",
        "event_type",
        "event_time",
        "platform",
    ]

    dicts = [
        "event_type",
        "region",
        "country",
        "language",
        "device_family",
        "device_type",
        "os_name",
    ]
    user_id = supabase.table("user_table_refined_v3").select("user_id").eq("user_index", refined_user_id).execute().data[0]["user_id"]
    user_response = supabase.table("user_table_refined_v3").select("amplitude_id, average_session_time, total_session_time, user_retention_30").eq("user_id", user_id).execute()
    user_response = user_response.data[0]
    amplitude_id = user_response["amplitude_id"]

    dict_mapping = {}
    for category in dicts:
        dict_response = (
            supabase.table(category).select(f"{category}, dict_{category}").execute()
        )
        mapping = {}
        for row in dict_response.data:
            key = row[category]
            value = row[f"dict_{category}"]
            mapping[key] = value
        dict_mapping[category] = mapping    
    
    batch_size = 1000
    offset = chunk * batch_size
    events_response = []

    events_batch = (
        supabase.table(TABLE_NAME)
        .select("*")
        .order("event_time", desc=False)
        .eq("amplitude_id", amplitude_id)
        .range(offset, offset + batch_size - 1)
        .execute()
    )
    if not events_batch.data:
        print(f"no chunk at rows {offset} to {offset + batch_size}")
        return f"no chunk at rows {offset} to {offset + batch_size}", 400
    events_response.extend(events_batch.data)
    
    print(f"processed rows {offset} to {offset + len(events_response)}")

    events = []
    for event in events_response:
        result = {}
        for category in categories:
            result[category] = event[category]

        for category in dicts:
            result[f"dict_{category}"] = dict_mapping.get(category).get(
                result[category]
            )

        result["average_session_time"] = user_response["average_session_time"]
        result["total_session_time"] = user_response["total_session_time"]
        result["user_retention_30"] = user_response["user_retention_30"]

        events.append(result)

    for event in events:
        if "." in event["event_time"]:
            date_part, ms_part = event["event_time"].split(".")
            ms_part = (ms_part + "000000")[:6]
            event["event_time"] = datetime.fromisoformat(f"{date_part}.{ms_part}")
        else:
            event["event_time"] = datetime.fromisoformat(event["event_time"])

    for i, event in enumerate(events):
        if i > 0:
            event["time_since_last"] = (
                event["event_time"] - events[i - 1]["event_time"]
            ).total_seconds()
            if event["time_since_last"] < 0:
                event["time_since_last"] = None
        else:
            event["time_since_last"] = None

        if i < len(events) - 1:
            event["time_to_next_event"] = (
                events[i + 1]["event_time"] - event["event_time"]
            ).total_seconds()
            if event["time_to_next_event"] < 0:
                event["time_to_next_event"] = None
        else:
            event["time_to_next_event"] = None

        for j in range(1, 5):
            if i - j >= 0:
                event[f"prev_{j}_event_type"] = events[i - j]["event_type"]
                event[f"dict_pet{j}"] = events[i - j]["dict_event_type"]
                event[f"time_since_last_{j}"] = (
                    event["event_time"] - events[i - j]["event_time"]
                ).total_seconds()
                if event[f"time_since_last_{j}"] < 0:
                    event[f"time_since_last_{j}"] = None
            else:
                event[f"prev_{j}_event_type"] = ""
                event[f"dict_pet{j}"] = None
                event[f"time_since_last_{j}"] = None

        if i < len(events) - 1:
            event["next_et"] = events[i + 1]["event_type"]
            event["dict_next_et"] = events[i + 1]["dict_event_type"]
        else:
            event["next_et"] = ""
            event["dict_next_et"] = None

    for event in events:
        event["event_time"] = event["event_time"].isoformat()

    return jsonify(events)

@app.route("/predict/single", methods=["POST"])
def predict():
    data = request.json  # Expecting JSON input
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Encode the input data
    encoded_features= encode_data(data, mode=1)

    # Convert to the appropriate tensor format
    encoded_features = encoded_features.unsqueeze(0)  # Add batch dimension if necessary

    # Forward pass through the model
    with torch.no_grad():
        candidate_events, time_prediction = model(encoded_features, edge_index=None, h_prev1=h_prev1, h_prev2=h_prev2)

    # Decode the output
    decoded_event_index = decode_event(candidate_events[0], model.event_embedding_layer)

    return jsonify({
        "predicted_event_index": decoded_event_index,
        "predicted_time": 10 ** (time_prediction.item()*10)  # Convert tensor to Python float
    })

if __name__ == "__main__":
    app.run(debug=True)
