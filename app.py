import os
from flask import Flask, jsonify
from supabase import create_client

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/", methods=["GET"])
def hello():
    return "API is running!"

# just to test the connection is working, you can go to this route
@app.route("/get_first_entry", methods=["GET"])
def get_first_entry():
    response = supabase.table("v2_federato_amplitude_data").select("*").limit(1).single().execute()
    data = response.data
    return jsonify(data)

def get

if __name__ == "__main__":
    app.run(debug=True)
