"""Read-only Streamlit dashboard for pipeline verification and visitor analytics."""

import sqlite3
import json
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "database" / "intelligent_face_tracker.sqlite3"
CONFIG_PATH = ROOT_DIR / "config.json"


@st.cache_resource(ttl=30, show_spinner=False)
def connect_mongodb():
    try:
        import pymongo

        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            database_config = json.load(config_file).get("database", {})
        client = pymongo.MongoClient(
            database_config.get("mongodb_uri", ""),
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
        )
        client.admin.command("ping")
        return client[database_config.get("database_name", "intelligent_face_tracker")]
    except Exception as error:
        st.session_state["mongodb_error"] = str(error)
        return None


def load_rows(query, parameters=()):
    if not DATABASE_PATH.exists():
        return []
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query, parameters)]


def load_mongo_rows(collection_name, query=None, limit=None, sort=None):
    mongo_db = connect_mongodb()
    if mongo_db is None:
        return None
    try:
        cursor = mongo_db[collection_name].find(query or {}, {"_id": 0})
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except Exception:
        return None


mongo_db = connect_mongodb()
data_source = "MongoDB"
mongo_counts = (0, 0)
mongo_fallback_reason = None
if mongo_db is not None:
    try:
        mongo_counts = (
            mongo_db["visitors"].count_documents({}),
            mongo_db["events"].count_documents({}),
        )
    except Exception:
        mongo_db = None
sqlite_counts = load_rows(
    "SELECT (SELECT COUNT(*) FROM visitors) AS visitors, "
    "(SELECT COUNT(*) FROM events) AS events"
)
local_has_data = bool(sqlite_counts and any(sqlite_counts[0].values()))
if mongo_db is None or (mongo_counts == (0, 0) and local_has_data):
    data_source = "SQLite fallback"
    if mongo_db is not None:
        mongo_fallback_reason = (
            "MongoDB is reachable but its visitors/events collections are empty."
        )
        st.session_state["mongodb_error"] = mongo_fallback_reason
    mongo_db = None
st.set_page_config(page_title="Intelligent Face Tracker", page_icon="👤", layout="wide")
st.title("Intelligent Face Tracker")
if data_source == "MongoDB":
    st.success("Data source: MongoDB (live)")
else:
    error_message = st.session_state.get("mongodb_error")
    if mongo_fallback_reason:
        st.warning(
            "MongoDB is connected but empty. Data source: local SQLite fallback."
        )
    else:
        st.warning("MongoDB is unavailable. Data source: local SQLite fallback.")
    if error_message:
        st.caption(f"MongoDB connection detail: {error_message}")


if mongo_db is not None:
    registered_count = mongo_db["visitors"].count_documents({})
    all_mongo_events = list(mongo_db["events"].find({}, {"created_at": 1, "video": 1, "_id": 0}))
    date_values = sorted(
        {row["created_at"][:10] for row in all_mongo_events if row.get("created_at")},
        reverse=True,
    )
    video_values = sorted({row["video"] for row in all_mongo_events if row.get("video")})
else:
    registered_rows = load_rows("SELECT COUNT(*) AS count FROM visitors")
    registered_count = registered_rows[0]["count"] if registered_rows else 0
    date_values = [
        row["event_date"]
        for row in load_rows(
            "SELECT DISTINCT substr(created_at, 1, 10) AS event_date "
            "FROM events WHERE created_at IS NOT NULL ORDER BY event_date DESC"
        )
    ]
    video_values = [
        row["video"]
        for row in load_rows("SELECT DISTINCT video FROM events ORDER BY video")
    ]

with st.sidebar:
    st.header("Filters")
    selected_date = st.selectbox(
        "Event date",
        ["All dates"] + date_values,
    )
    selected_video = st.selectbox(
        "Video",
        ["All videos"] + video_values,
    )
    visitor_search = st.text_input(
        "Search visitor ID",
        placeholder="e.g. VISITOR_0006",
    )
    selected_event = st.selectbox(
        "Event type",
        ["All events", "ENTRY", "EXIT"],
    )

conditions = []
parameters = []
if selected_date != "All dates":
    if mongo_db is None:
        conditions.append("substr(created_at, 1, 10) = ?")
        parameters.append(selected_date)
if selected_video != "All videos":
    if mongo_db is None:
        conditions.append("video = ?")
        parameters.append(selected_video)
if visitor_search.strip() and mongo_db is None:
    conditions.append("visitor_id = ?")
    parameters.append(visitor_search.strip())
if selected_event != "All events" and mongo_db is None:
    conditions.append("event_type = ?")
    parameters.append(selected_event)
where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

if mongo_db is not None:
    mongo_query = {}
    if selected_date != "All dates":
        mongo_query["created_at"] = {"$regex": f"^{selected_date}"}
    if selected_video != "All videos":
        mongo_query["video"] = selected_video
    if visitor_search.strip():
        mongo_query["visitor_id"] = visitor_search.strip()
    if selected_event != "All events":
        mongo_query["event_type"] = selected_event
    filtered_events = list(mongo_db["events"].find(mongo_query, {"_id": 0}))
    event_rows = sorted(
        filtered_events,
        key=lambda row: row.get("created_at", ""),
        reverse=True,
    )[:200]
    summary = {
        "events": len(filtered_events),
        "unique_visitors": len({row.get("visitor_id") for row in filtered_events}),
        "entries": sum(row.get("event_type") == "ENTRY" for row in filtered_events),
        "exits": sum(row.get("event_type") == "EXIT" for row in filtered_events),
    }
else:
    summary_rows = load_rows(
        f"""
        SELECT COUNT(*) AS events, COUNT(DISTINCT visitor_id) AS unique_visitors,
        SUM(CASE WHEN event_type = 'ENTRY' THEN 1 ELSE 0 END) AS entries,
        SUM(CASE WHEN event_type = 'EXIT' THEN 1 ELSE 0 END) AS exits
        FROM events {where_clause}
        """,
        parameters,
    )
    summary = summary_rows[0] if summary_rows else {
        "events": 0, "unique_visitors": 0, "entries": 0, "exits": 0,
    }

metrics = st.columns(5)
metrics[0].metric("Registered visitors", registered_count)
metrics[1].metric("Unique in selection", summary["unique_visitors"] or 0)
metrics[2].metric("Entries", summary["entries"] or 0)
metrics[3].metric("Exits", summary["exits"] or 0)
metrics[4].metric("Event balance", (summary["entries"] or 0) - (summary["exits"] or 0))

if mongo_db is None:
    event_rows = load_rows(
        f"SELECT video, frame, timestamp, visitor_id, event_type, crop_path, created_at "
        f"FROM events {where_clause} ORDER BY id DESC LIMIT 200",
        parameters,
    )

st.subheader("Recent events")
if not event_rows:
    st.info("No events match the selected filters.")
else:
    st.dataframe(
        [
            {
                "Video": row["video"],
                "Frame": row["frame"],
                "Timestamp (s)": row["timestamp"],
                "Visitor ID": row["visitor_id"],
                "Event": row["event_type"],
                "Crop path": row["crop_path"],
                "Created": row["created_at"],
            }
            for row in event_rows
        ],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Use the visitor ID search and Event type filter to query who entered or exited. "
    "Database paths are stored relative to the project."
)
