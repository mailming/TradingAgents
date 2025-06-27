"""
Simple API Server for TradingAgents Analysis Results

This creates a simple Flask API to serve the JSON trading analysis results
for frontend consumption. Perfect for demonstrating how to integrate with
web applications and dashboards.

Features:
- REST API endpoints for analysis results
- CORS enabled for frontend access
- Automatic file discovery
- Error handling
- JSON response formatting

Usage: python simple_api_server.py
Access: http://localhost:5000/api/analyses/latest
"""

import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import glob

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configuration
ANALYSIS_DIR = Path("analysis_results")
JSON_DIR = ANALYSIS_DIR / "json"
HTML_DIR = ANALYSIS_DIR / "html"
MARKDOWN_DIR = ANALYSIS_DIR / "markdown"

@app.route('/')
def index():
    """API documentation"""
    return {
        "name": "TradingAgents Analysis API",
        "version": "1.0",
        "description": "REST API for trading analysis results",
        "endpoints": {
            "GET /api/analyses": "List all analyses",
            "GET /api/analyses/latest": "Get latest analysis",
            "GET /api/analyses/<analysis_id>": "Get specific analysis",
            "GET /api/analyses/ticker/<ticker>": "Get analyses for ticker",
            "GET /api/dashboard": "Frontend dashboard",
            "POST /api/run-analysis": "Run new analysis (if available)"
        },
        "powered_by": "financialdatasets.ai + Claude Haiku"
    }

@app.route('/api/analyses')
def get_all_analyses():
    """Get list of all analyses"""
    try:
        json_files = list(JSON_DIR.glob("*.json"))
        analyses = []
        
        for file_path in sorted(json_files, reverse=True):  # Latest first
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    analyses.append({
                        "analysis_id": data["analysis_metadata"]["analysis_id"],
                        "ticker": data["analysis_metadata"]["ticker"],
                        "date": data["analysis_metadata"]["analysis_date"],
                        "timestamp": data["analysis_metadata"]["timestamp"],
                        "decision": data["final_decision"]["recommendation"],
                        "confidence": data["final_decision"]["confidence_level"],
                        "duration": data["analysis_metadata"]["duration_seconds"],
                        "filename": file_path.name
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        return jsonify({
            "success": True,
            "count": len(analyses),
            "analyses": analyses
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analyses/latest')
def get_latest_analysis():
    """Get the most recent analysis"""
    try:
        json_files = list(JSON_DIR.glob("*.json"))
        if not json_files:
            return jsonify({
                "success": False,
                "error": "No analyses found"
            }), 404
        
        # Get the most recent file
        latest_file = max(json_files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            "success": True,
            "analysis": data,
            "file_info": {
                "filename": latest_file.name,
                "created": datetime.fromtimestamp(os.path.getmtime(latest_file)).isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analyses/<analysis_id>')
def get_analysis_by_id(analysis_id):
    """Get specific analysis by ID"""
    try:
        json_files = list(JSON_DIR.glob(f"*_{analysis_id}.json"))
        
        if not json_files:
            return jsonify({
                "success": False,
                "error": f"Analysis {analysis_id} not found"
            }), 404
        
        file_path = json_files[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            "success": True,
            "analysis": data,
            "file_info": {
                "filename": file_path.name,
                "created": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analyses/ticker/<ticker>')
def get_analyses_by_ticker(ticker):
    """Get all analyses for a specific ticker"""
    try:
        ticker = ticker.upper()
        json_files = list(JSON_DIR.glob(f"{ticker}_*.json"))
        analyses = []
        
        for file_path in sorted(json_files, reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    analyses.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if not analyses:
            return jsonify({
                "success": False,
                "error": f"No analyses found for ticker {ticker}"
            }), 404
        
        return jsonify({
            "success": True,
            "ticker": ticker,
            "count": len(analyses),
            "analyses": analyses
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/dashboard')
def serve_dashboard():
    """Serve the frontend dashboard"""
    try:
        return send_from_directory('.', 'frontend_demo.html')
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Dashboard not found",
            "details": str(e)
        }), 404

@app.route('/api/reports/<format>/<filename>')
def serve_report(format, filename):
    """Serve report files (HTML, JSON, Markdown)"""
    try:
        if format == 'json':
            return send_from_directory(JSON_DIR, filename)
        elif format == 'html':
            return send_from_directory(HTML_DIR, filename)
        elif format == 'markdown':
            return send_from_directory(MARKDOWN_DIR, filename)
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported format: {format}"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

@app.route('/api/run-analysis', methods=['POST'])
def run_new_analysis():
    """Run a new analysis (if analysis system is available)"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'TSLA').upper()
        
        # Check if the analysis capture system is available
        try:
            from trading_analysis_capture import TradingAnalysisCapture
            
            # Run new analysis
            capture = TradingAnalysisCapture()
            results = capture.run_analysis_and_capture(ticker)
            file_paths = capture.save_results(results)
            
            return jsonify({
                "success": True,
                "message": f"Analysis completed for {ticker}",
                "analysis": results,
                "files": file_paths
            })
            
        except ImportError:
            return jsonify({
                "success": False,
                "error": "Analysis system not available",
                "message": "The trading analysis capture system is not accessible from this API"
            }), 503
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/stats')
def get_system_stats():
    """Get system statistics"""
    try:
        json_files = list(JSON_DIR.glob("*.json"))
        html_files = list(HTML_DIR.glob("*.html"))
        md_files = list(MARKDOWN_DIR.glob("*.md"))
        
        # Count by ticker
        ticker_counts = {}
        total_analyses = 0
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ticker = data["analysis_metadata"]["ticker"]
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                    total_analyses += 1
            except:
                continue
        
        return jsonify({
            "success": True,
            "statistics": {
                "total_analyses": total_analyses,
                "total_json_files": len(json_files),
                "total_html_files": len(html_files),
                "total_markdown_files": len(md_files),
                "ticker_breakdown": ticker_counts,
                "data_sources": ["financialdatasets.ai"],
                "ai_models": ["claude-3-haiku-20240307"],
                "system_status": "operational"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def create_directories():
    """Create necessary directories if they don't exist"""
    for directory in [ANALYSIS_DIR, JSON_DIR, HTML_DIR, MARKDOWN_DIR]:
        directory.mkdir(exist_ok=True)

if __name__ == '__main__':
    create_directories()
    
    print("🚀 TradingAgents Analysis API Server Starting...")
    print("=" * 50)
    print(f"📊 Analysis Directory: {ANALYSIS_DIR}")
    print(f"📁 JSON Files: {len(list(JSON_DIR.glob('*.json')))}")
    print(f"📄 HTML Files: {len(list(HTML_DIR.glob('*.html')))}")
    print(f"📝 Markdown Files: {len(list(MARKDOWN_DIR.glob('*.md')))}")
    print()
    print("🌐 API Endpoints:")
    print("   GET  http://localhost:5000/                    - API documentation")
    print("   GET  http://localhost:5000/api/analyses        - List all analyses")
    print("   GET  http://localhost:5000/api/analyses/latest - Get latest analysis")
    print("   GET  http://localhost:5000/api/dashboard       - Frontend dashboard")
    print("   POST http://localhost:5000/api/run-analysis    - Run new analysis")
    print()
    print("🎯 Ready for frontend integration!")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 