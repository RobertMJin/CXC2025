# Project Title
CXC2025-Federato Thingy

## Description
We cooking

## Installation
Instructions on how to install and set up your project.

Create python venv

python3 -m venv .vevn
source .venv/bin/activate
pip install requirements.txt

Create a .env file with the following items so that you can access the DB if you are using data_reader.ipynb:
USER=
PASSWORD=
HOST=db.lctgsjrpfpuivoouotjy.supabase.co
PORT=5432
DBNAME=postgres

DB querying can be accessed using the DB URL:
DATABASE_URL=

To run the Docker container without destroying the existing Postgres database, follow these steps:
1. Open a terminal and navigate to the directory where your `docker-compose.yaml` file is located.
2. Run the command `docker-compose up -d` to start the containers in the background.
3. To stop the containers, run the command `docker-compose down`.