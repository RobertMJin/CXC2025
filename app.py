from datetime import datetime
import os
from flask import Flask, jsonify, request
from supabase import create_client, Client

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE_NAME = "v2_federato_amplitude_data"
# TABLE_NAME = "v2024_1_federato_amplitude_data"

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

    user_response = supabase.table("user_table").select("amplitude_id, average_session_time, total_session_time, user_retention_30").eq("user_id", user_id).execute()
    user_response = user_response.data[0]
    amplitude_id = user_response["amplitude_id"]

    dict_mapping = {}
    for category in dicts:
        dict_response = supabase.table(category).select(f"{category}, dict_{category}").execute()
        mapping = {}
        for row in dict_response.data:
            key = row[category]
            value = row[f"dict_{category}"]
            mapping[key] = value
        dict_mapping[category] = mapping    
    
    events_response = supabase.table(TABLE_NAME).select("*").eq("amplitude_id", amplitude_id).order("event_time", desc=False).execute()
    
    events = []
    for event in events_response.data:
        result = {}
        for category in categories:
            result[category] = event[category]
        
        for category in dicts:
            result[f"dict_{category}"] = dict_mapping.get(category).get(result[category])

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
            event["time_since_last"] = (event["event_time"] - events[i-1]["event_time"]).total_seconds()
            if event["time_since_last"] < 0:
                event["time_since_last"] = None
        else:
            event["time_since_last"] = None
        
        if i < len(events) - 1:
            event["time_to_next_event"] = (events[i+1]["event_time"] - event["event_time"]).total_seconds()
            if event["time_to_next_event"] < 0:
                event["time_to_next_event"] = None
        else:
            event["time_to_next_event"] = None

        for j in range(1, 5):
            if i - j >= 0:
                event[f"prev_{j}_event_type"] = events[i - j]["event_type"]
                event[f"dict_pet{j}"] = events[i - j]["dict_event_type"]
                event[f"time_since_last_{j}"] = (event["event_time"] - events[i-j]["event_time"]).total_seconds()
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

    results = [get_user_sessions(user_id).json for user_id in range(start_id, end_id + 1)]
    
    return jsonify(results)

# def get_dict_values(): in progress
#     dicts = [
#         "event_type",
#         "region",
#         "country",
#         "language",
#         "device_family",
#         "device_type",
#         "os_name",
#     ]

#     dict_mapping = {}
#     for category in dicts:
#         dict_response = supabase.table(category).select(f"{category}, dict_{category}").execute()
#         mapping = {}
#         for row in dict_response.data:
#             key = row[category]
#             value = row[f"dict_{category}"]
#             mapping[key] = value
#         dict_mapping[category] = mapping  

#     events = []
#     for event in events_response.data:
#         result = {}
#         for category in categories:
#             result[category] = event[category]
        
#         for category in dicts:
#             result[f"dict_{category}"] = dict_mapping.get(category).get(result[category])

#         events.append(result)


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

    if not app_type or not region or not country or not device_family or not device_type or not os_name or not platform:
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

@app.route('/get_profile', methods=['GET'])
def get_profile():
    return jsonify(profile)

# @app.route("/create_session", methods=["GET"]) in progress
# def create_user_session():
#     events = []
#     for i in range(5):
#         events.append(request.form.get(f"event_{i}"))
#     session = profile
#     return


if __name__ == "__main__":
    app.run(debug=True)
