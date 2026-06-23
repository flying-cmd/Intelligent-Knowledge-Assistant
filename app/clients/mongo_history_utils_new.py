# Standard-library imports.
import os
# Logging utilities.
import logging
# Type hints for readability and editor support.
from typing import List, Dict, Any, Optional
# Timestamp generation for stored messages.
from datetime import datetime
# Core PyMongo driver types. `ASCENDING` is used for sorting and indexes.
from pymongo import MongoClient, ASCENDING
# MongoDB default primary-key type.
from bson import ObjectId
# Load environment variables from `.env`.
from dotenv import load_dotenv

# Load `.env` so `os.getenv` can read the configuration.
load_dotenv()


class HistoryMongoTool:
    """
    MongoDB helper for reading and writing conversation history.
    """

    def __init__(self):
        """
        Initialize the MongoDB connection, database, collection, and indexes.
        """
        try:
            # Read MongoDB settings from the environment.
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            # Create the MongoDB client and select the target collection.
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # Create the compound index used by session-history queries.
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise


def clear_history(session_id: str) -> int:
    """
    Delete all conversation-history records for a session.
    :param session_id: Session identifier
    :return: Number of deleted documents, or 0 on failure
    """
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None
) -> str:
    """
    Insert or update a single chat-history record in MongoDB.
    :param session_id: Session identifier
    :param role: Message role, usually `user` or `assistant`
    :param text: Core message content
    :param rewritten_query: Rewritten query text
    :param item_names: Related product names
    :param message_id: Existing message ID for update mode
    :return: Inserted or updated record identifier
    """
    # Store timestamps as seconds for later sorting and filtering.
    ts = datetime.now().timestamp()

    # Document payload to insert or update.
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document}
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    Batch-update `item_names` for selected message records.
    Only records with empty or missing `item_names` are updated.
    :param ids: Message ID list as strings
    :param item_names: New product-name list
    :return: Number of updated documents, or 0 on failure
    """
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None}
                ]
            },
            {"$set": {"item_names": item_names}}
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query the most recent N conversation messages for a session.
    Results are returned in chronological order so they can be passed to an LLM.
    :param session_id: Session identifier
    :param limit: Maximum number of records to return
    :return: List of raw message dictionaries
    """
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}

        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        messages = list(cursor)

        return messages
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []


# Module-level singleton to avoid creating repeated MongoDB connections.
_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    Return the `HistoryMongoTool` singleton using lazy initialization.
    :return: Shared `HistoryMongoTool` instance
    """
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


# Try to initialize early at module load for faster first use.
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    # Keep lazy-load fallback behavior if eager initialization fails.
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")

# Local smoke test.
if __name__ == "__main__":
    sid = "000015_hybrid"
    save_chat_message(sid, "user", "Hello (Hybrid)")
    save_chat_message(sid, "assistant", "Hello! I am an assistant built on native Mongo + LangChain objects.")
    save_chat_message(sid, "user", "How do I replace the battery in this multimeter?", item_names=["Hybrid Multimeter"])

    print("--- Querying LangChain object records ---")
    messages = get_recent_messages(sid, limit=5)
    print(f"Number of records found: {len(messages)}")
    for m in messages:
        print(f" {m}  ")
