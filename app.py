from datetime import datetime
import os
from flask import Flask, jsonify
from supabase import create_client, Client

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# TABLE_NAME = "v2_federato_amplitude_data"
TABLE_NAME = "v2024_1_federato_amplitude_data"

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

    user_response = supabase.table("user_table").select("amplitude_id").eq("user_id", user_id).execute()
    amplitude_id = user_response.data[0]["amplitude_id"]

    dict_mapping = {}
    for category in dicts:
        dict_response = supabase.table(category).select(f"{category}, dict_{category}").execute()
        mapping = {}
        for row in dict_response.data:
            key = row[category]
            value = row[f"dict_{category}"]
            mapping[key] = value
        dict_mapping[category] = mapping    
    
    events_response = supabase.table(TABLE_NAME).select("*").eq("amplitude_id", amplitude_id).execute()
    
    events = []
    for event in events_response.data:
        result = {}
        for category in categories:
            result[category] = event[category]
        
        for category in dicts:
            result[f"dict_{category}"] = dict_mapping.get(category).get(result[category])

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
                event[f"time_since_last_{j}"] = (event["event_time"] - events[i-j]["event_time"]).total_seconds()
                if event[f"time_since_last_{j}"] < 0:
                    event[f"time_since_last_{j}"] = None
            else:
                event[f"prev_{j}_event_type"] = ""
                event[f"time_since_last_{j}"] = None

        if i < len(events) - 1:
            event["next_event_type"] = events[i + 1]["event_type"]
            event["dict_next_event"] = events[i + 1]["dict_event_type"]
        else:
            event["next_event_type"] = ""
            event["dict_next_event"] = None

    for event in events:
        event["event_time"] = event["event_time"].isoformat()
    
    return events



if __name__ == "__main__":
    app.run(debug=True)
