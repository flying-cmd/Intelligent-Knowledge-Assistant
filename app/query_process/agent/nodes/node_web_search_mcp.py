import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse
from app.core.logger import logger

async def mcp_call(query):
    """
    Core async function for calling the Bailian MCP search service.
    """
    search_mcp = MCPServerSse(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300,
            "sse_read_timeout": 300
        }
    )

    try:
        logger.info(f"[MCP] Connecting to the Bailian WebSearch service: {mcp_config.mcp_base_url}")
        await search_mcp.connect()
        
        logger.info(f"[MCP] Connected successfully, calling tool 'bailian_web_search' with query: {query}")
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search", 
            arguments={"query": query, "count": 5}
        )
        logger.info("[MCP] Tool call completed and returned a result")
        return result
        
    except Exception as e:
        logger.error(f"[MCP] Exception during the tool call: {e}", exc_info=True)
        return None
        
    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """
    Synchronous LangGraph node that performs MCP-based web search.
    """
    logger.info("---node_web_search_mcp started---")
    
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 2. Resolve the query text.
    query = state.get("rewritten_query", "")
    if not query:
        query = state.get("original_query", "")
        
    docs = []
    
    # 3. Execute the search.
    if query:
        try:
            logger.info(f"Starting async MCP call for query: {query}")
            result = asyncio.run(mcp_call(query))
            
            # 4. Parse the result.
            if result and not result.isError and result.content:
                raw_text = result.content[0].text
                try:
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []
                    
                    logger.info(f"MCP returned {len(pages)} raw pages")
                    
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()
                        
                        if not snippet:
                            continue
                            
                        docs.append({"title": title, "url": url, "snippet": snippet})
                        
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse MCP JSON response: {raw_text[:100]}...")
            else:
                if result and result.isError:
                    logger.error(f"MCP returned an error result: {result}")
                else:
                    logger.warning("MCP returned an empty or invalid result")

            logger.info(f"Structured web-search result count: {len(docs)}")
            
        except Exception as e:
            logger.error(f"MCP web-search node failed: {e}", exc_info=True)
    else:
        logger.warning("Query text is empty, skipping MCP search")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    
    logger.info("---node_web_search_mcp finished---")
    
    if docs:
        return {"web_search_docs": docs}
    return {}


if __name__ == '__main__':
    print("\n" + "="*50)
    print(">>> Starting local test for node_web_search_mcp")
    print("="*50)
    
    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "Under the factory default settings, how should the HAK 180 control panel be configured so foil transfers only to the top 50 mm-170 mm area of the paper?",
        "is_stream": False
    }

    try:
        result_state = node_web_search_mcp(test_state)

        print("\n" + "="*50)
        print(">>> Test result summary:")
        search_results = result_state.get('web_search_docs', [])
        print(f"Search result count: {len(search_results)}")
        if search_results:
            print("First result preview:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("No search results were returned")
        print("="*50)
        
    except Exception as e:
        logger.exception(f"Uncaught exception during local test: {e}")
