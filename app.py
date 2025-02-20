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

# just to test the connection is working, you can go to this route
@app.route("/get_first_entry", methods=["GET"])
def get_first_entry():
    response = supabase.table(TABLE_NAME).select("*").limit(1).single().execute()
    data = response.data
    return jsonify(data)

# DATA

# need to get all unique entries for each categorical variables (country, language, device_type, etc.)
@app.route("/unique_categories", methods=["GET"])
def get_unique_categories():
    categorical_variables = ["app", "region", "country", "language", "device_family", "device_type", "os_name", "platform"]
    result = {}

    for column in categorical_variables:
        response = (
            supabase.rpc("count_distinct", {"column_name": column}).execute()
        )
        if response.data:
            result[column] = response.data[0]["count"]  # Extract the unique count

    return jsonify(result)



# USERS

# returns a json of data from a specifc user's session
@app.route("model/search/user/<user_id>")
def user_session(user_id):
    response = supabase.table(TABLE_NAME)\
        .select("amplitude_id, app, country, device_family, device_type, language, os_name, platform, region, session_id, user_properties")\
        .eq("user_id", user_id)\
        .execute()

    if response.error:
        return jsonify({"error": response.error.message}), 400
    
    return jsonify(response.data), 200
    

# CALCULATIONS
# TODO



if __name__ == "__main__":
    app.run(debug=True)