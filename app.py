from datetime import datetime
import os, time, json
from flask import Flask, jsonify, request
from supabase import create_client, Client
from flask_cors import CORS
import torch
from model import encode_data, GATMinGRU, decode_event
import dotenv
from openai import OpenAI

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
            "expose_headers": ["Content-Type", "Authorization"],
        }
    },
)

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# model = GATMinGRU(input_size=166, hidden_size=256, event_embedding_size=16, gat_heads=2)
# model.load_state_dict(torch.load("/Users/fahmiomer/CXC2025/checkpoint_epoch_6.pth"))
# model.eval()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE_NAME = "v2_federato_amplitude_data"
# TABLE_NAME = "v2024_1_federato_amplitude_data"

model = GATMinGRU(input_size=166, hidden_size=512, event_embedding_size=16, gat_heads=4)
checkpoint = torch.load("./model.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
h_prev1 = torch.zeros(1, 512)
h_prev2 = torch.zeros(1, 512)


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


@app.route("/create-session", methods=["GET", "POST", "OPTIONS"])
def create_user_session():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    try:
        data = request.get_json()
        print("Received data:", data)  # Debug log

        user_id = data["userId"]
        events = data["events"]

        print(f"Looking up user_id: {user_id}")  # Debug log

        # Try direct lookup in user_table first
        try:
            user = (
                supabase.table("user_table")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if not user.data:
                return jsonify({"error": "User not found in any table"}), 404
            user = user.data[0]

        except Exception as e:
            print(f"Error during user lookup: {str(e)}")  # Debug log
            return jsonify({"error": f"User lookup failed: {str(e)}"}), 500

        # Get event type dictionary mapping
        dict_response = (
            supabase.table("event_type").select("event_type, dict_event_type").execute()
        )
        event_type_mapping = {
            row["event_type"]: row["dict_event_type"] for row in dict_response.data
        }

        # Initialize event_json with default values
        event_json = {
            "dict_next_et": None,
            "next_et": None,
            "time_since_last": 0,
            "time_to_next_event": 0,
            "app": 591532,  # Set default app value
            "amplitude_id": user.get("amplitude_id", ""),
            "average_session_time": user.get("average_session_time", 0),
            "country": user.get("country", "Unknown"),
            "device_family": user.get("device_family", "Unknown"),
            "device_type": user.get("device_type", "Unknown"),
            "language": user.get("language", "Unknown"),
            "event_time": user.get("event_time", datetime.now().isoformat()),
            "os_name": user.get("os_name", "Unknown"),
            "region": user.get("region", "Unknown"),
            "platform": user.get("platform", "Web"),
            "event_type": events[4],
            "dict_event_type": event_type_mapping.get(
                events[4], 0
            ),  # Default to 0 if not found
            "session_id": user.get("session_id", 0),
            "dict_region": 0,
            "dict_country": 0,
        }

        # Set previous events and their dictionary values
        for i in range(1, 5):
            prev_event = events[4 - i]
            event_json[f"prev_{i}_event_type"] = prev_event
            event_json[f"dict_pet{i}"] = event_type_mapping.get(
                prev_event, 0
            )  # Default to 0 if not found
            event_json[f"time_since_last_{i}"] = 0

        return jsonify(event_json)

    except Exception as e:
        print(f"Error in create_user_session: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500


@app.route("/user-data", methods=["POST"])
def get_user_data():
    data = request.get_json()
    user_id = data["userId"]

    user = supabase.table("user_table").select("*").eq("user_id", user_id).execute()
    if user.data:
        user = user.data[0]
        return jsonify(user)
    return "no user", 400


@app.route(
    "/model/search/user_chunk/<int:refined_user_id>/<int:chunk>", methods=["GET"]
)
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
    user_id = (
        supabase.table("user_table_refined_v3")
        .select("user_id")
        .eq("user_index", refined_user_id)
        .execute()
        .data[0]["user_id"]
    )
    user_response = (
        supabase.table("user_table_refined_v3")
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

    # Set default values for missing or empty fields
    if not data.get("platform"):
        data["platform"] = "Web"  # Default platform

    if not data.get("device_type"):
        data["device_type"] = "Unknown"

    if not data.get("os_name"):
        data["os_name"] = "Unknown"

    if not data.get("language"):
        data["language"] = "Unknown"

    if not data.get("device_family"):
        data["device_family"] = "Unknown"

    try:
        # Encode the input data
        encoded_features = encode_data(data, mode=1)

        # Convert to the appropriate tensor format
        encoded_features = encoded_features.unsqueeze(
            0
        )  # Add batch dimension if necessary

        # Forward pass through the model
        with torch.no_grad():
            candidate_events, time_prediction = model(
                encoded_features, edge_index=None, h_prev1=h_prev1, h_prev2=h_prev2
            )
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    # Decode the output
    decoded_event_index = decode_event(candidate_events[0], model.event_embedding_layer)
    decoded_event_index2 = decode_event(candidate_events[1], model.event_embedding_layer)
    decoded_event_index3 = decode_event(candidate_events[2], model.event_embedding_layer)
    return jsonify(
        {
            "predicted_event_index": decoded_event_index,
            "predicted_event_index2": decoded_event_index2,
            "predicted_event_index3": decoded_event_index3,
            "predicted_time": 10 ** (time_prediction.item() * 10),
        }
    )
    

    
@app.route("/llminsights", methods=["POST"])
class Solution:
    suggestion: str
    explanation: str
class Response:
    solutions: list[Solution]
    summary: str
    
def llminsights():
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "system", "content": "You are a data analyst for an online platform that wants to help increase user retention, mainly through reducing churn events from occurring."},
            {f"role": "user", "content": "Explain how recommended actions can be surfaced to users in real-time at decisive moments (e.g., when they are about to exit the platform). Suggest strategies to encourage longer daily usage and improve feature adoption based \
                on user behavior data. The next 3 predicted events for the current event {event_type} are:\n 1) {predicted_event_index1},  {probability1}, 2) {predicted_event_index2},  {probability2}, and 3) {predicted_event_index3},  {probability3}. This user commonly uses the following features: {features} and has a high probability of churning at these events {churn_events}."},
        ],
        response_format=Response
    )

    return jsonify({
       "suggestion": completion.choices[0].message
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
