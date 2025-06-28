from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
import uuid
from datetime import datetime
from pathlib import Path


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Look at news from EODHD, and finnhub to be comprehensive. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        news_data_json = {}

        # Capture news feed data from tool calls and save to JSON
        if len(result.tool_calls) > 0:
            # Import here to avoid circular imports
            try:
                from tradingagents.dataflows.json_export_utils import create_exporter
                
                news_data_captured = {
                    "finnhub_news": [],
                    "google_news": [],
                    "reddit_news": [],
                    "global_news": [],
                    "analysis_metadata": {
                        "ticker": ticker,
                        "analysis_date": current_date,
                        "timestamp": datetime.now().isoformat(),
                        "analyst_type": "news_analyst",
                        "data_source": "multiple_news_sources",
                        "tools_used": [tool.name for tool in tools]
                    }
                }
                
                # Execute tool calls and capture news results
                for tool_call in result.tool_calls:
                    try:
                        # Find the corresponding tool
                        tool_func = None
                        for tool in tools:
                            if tool.name == tool_call["name"]:
                                tool_func = tool
                                break
                        
                        if tool_func:
                            # Execute the tool and capture result
                            tool_result = tool_func.invoke(tool_call["args"])
                            
                            # Categorize news by source
                            news_entry = {
                                "tool_used": tool_call["name"],
                                "parameters": tool_call["args"],
                                "timestamp": datetime.now().isoformat(),
                                "data": None
                            }
                            
                            # Process different types of news data
                            if hasattr(tool_result, 'to_dict'):
                                news_entry["data"] = tool_result.to_dict()
                            elif isinstance(tool_result, list):
                                news_entry["data"] = tool_result
                            elif isinstance(tool_result, dict):
                                news_entry["data"] = tool_result
                            else:
                                news_entry["data"] = str(tool_result)
                            
                            # Categorize by news source
                            if "finnhub" in tool_call["name"].lower():
                                news_data_captured["finnhub_news"].append(news_entry)
                            elif "google" in tool_call["name"].lower():
                                news_data_captured["google_news"].append(news_entry)
                            elif "reddit" in tool_call["name"].lower():
                                news_data_captured["reddit_news"].append(news_entry)
                            elif "global" in tool_call["name"].lower():
                                news_data_captured["global_news"].append(news_entry)
                            else:
                                # Generic news category
                                if "other_news" not in news_data_captured:
                                    news_data_captured["other_news"] = []
                                news_data_captured["other_news"].append(news_entry)
                    
                    except Exception as e:
                        print(f"Warning: Could not capture news from tool {tool_call['name']}: {e}")
                        news_data_captured[f"{tool_call['name']}_error"] = str(e)
                
                # Calculate summary statistics
                total_news_items = (
                    len(news_data_captured["finnhub_news"]) +
                    len(news_data_captured["google_news"]) +
                    len(news_data_captured["reddit_news"]) +
                    len(news_data_captured["global_news"])
                )
                
                # Save news data to JSON using ZZSheep exporter
                try:
                    exporter = create_exporter()
                    analysis_id = str(uuid.uuid4())[:8]
                    filename = f"{ticker}_{current_date}_{analysis_id}_news_feed.json"
                    
                    json_path = exporter.save_analysis_results(
                        news_data_captured,
                        ticker=ticker,
                        analysis_type="news_feed_capture",
                        custom_filename=filename
                    )
                    
                    news_data_json = {
                        "data_file": str(json_path),
                        "captured_at": datetime.now().isoformat(),
                        "data_summary": {
                            "total_news_items": total_news_items,
                            "finnhub_articles": len(news_data_captured["finnhub_news"]),
                            "google_articles": len(news_data_captured["google_news"]),
                            "reddit_posts": len(news_data_captured["reddit_news"]),
                            "global_articles": len(news_data_captured["global_news"]),
                            "tools_executed": len(result.tool_calls),
                            "sources_used": [tool_call["name"] for tool_call in result.tool_calls]
                        }
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save news data to JSON: {e}")
                    news_data_json = {"error": f"Failed to save: {e}"}
                    
            except ImportError as e:
                print(f"Warning: Could not import JSON export utility: {e}")
                news_data_json = {"error": "JSON export utility not available"}

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
            "news_data_json": news_data_json,
        }

    return news_analyst_node
