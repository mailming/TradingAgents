"""
Intelligent Model Router for TradingAgents

This module automatically routes requests to the most appropriate AI model:
- Claude Haiku (3.5) for simple, fast tasks
- Claude Sonnet (3.5) for complex reasoning and analysis
- Dynamic model selection based on task complexity
- Cost and performance optimization

Author: TradingAgents AI Optimization Team
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from ..adapters.anthropic_direct import DirectChatAnthropic

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"           # Simple data extraction, formatting
    MODERATE = "moderate"       # Basic analysis, summaries
    COMPLEX = "complex"         # Deep reasoning, multi-step analysis
    CRITICAL = "critical"       # High-stakes decisions, complex debates


class ModelType(Enum):
    """Available model types"""
    HAIKU = "claude-3-5-haiku-20241022"      # Fast, cost-effective
    SONNET = "claude-3-5-sonnet-20241022"    # Advanced reasoning


@dataclass
class ModelMetrics:
    """Metrics for model performance tracking"""
    model_name: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None


@dataclass
class TaskClassification:
    """Result of task complexity classification"""
    complexity: TaskComplexity
    confidence: float
    reasoning: str
    recommended_model: ModelType
    estimated_tokens: int


class TaskComplexityAnalyzer:
    """Analyzes task complexity to determine optimal model"""
    
    def __init__(self):
        """Initialize the complexity analyzer"""
        # Patterns for different complexity levels
        self.simple_patterns = [
            r"extract.*price",
            r"format.*data",
            r"summarize.*in.*words",
            r"current.*value",
            r"latest.*update",
            r"simple.*calculation",
            r"basic.*information",
            r"quick.*summary"
        ]
        
        self.moderate_patterns = [
            r"analyze.*trend",
            r"compare.*performance",
            r"evaluate.*metrics",
            r"assess.*risk",
            r"technical.*analysis",
            r"market.*overview",
            r"financial.*summary"
        ]
        
        self.complex_patterns = [
            r"investment.*recommendation",
            r"detailed.*analysis",
            r"multi.*factor.*analysis",
            r"comprehensive.*review",
            r"strategic.*planning",
            r"debate.*decision",
            r"complex.*reasoning",
            r"risk.*assessment.*with",
            r"fundamental.*analysis.*including"
        ]
        
        self.critical_patterns = [
            r"final.*investment.*decision",
            r"buy.*sell.*hold.*recommendation",
            r"portfolio.*allocation",
            r"risk.*management.*strategy",
            r"investment.*thesis",
            r"trading.*strategy",
            r"high.*stakes.*decision"
        ]
        
        # Keywords that indicate complexity
        self.complexity_keywords = {
            TaskComplexity.SIMPLE: [
                "format", "extract", "current", "latest", "simple", "quick", "basic"
            ],
            TaskComplexity.MODERATE: [
                "analyze", "compare", "evaluate", "assess", "technical", "summary", "overview"
            ],
            TaskComplexity.COMPLEX: [
                "detailed", "comprehensive", "multi-factor", "strategic", "debate", 
                "reasoning", "investigation", "research"
            ],
            TaskComplexity.CRITICAL: [
                "decision", "recommendation", "buy", "sell", "hold", "portfolio", 
                "investment", "trading", "final", "strategy"
            ]
        }
    
    def classify_task(self, messages: List[BaseMessage], context: Dict[str, Any] = None) -> TaskClassification:
        """
        Classify task complexity based on messages and context
        
        Args:
            messages: List of messages in the conversation
            context: Additional context about the task
            
        Returns:
            TaskClassification with complexity assessment
        """
        # Combine all message content
        full_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
        full_text = full_text.lower()
        
        # Get context information
        context = context or {}
        agent_type = context.get('agent_type', '')
        task_stage = context.get('task_stage', '')
        
        # Calculate complexity scores
        complexity_scores = {
            TaskComplexity.SIMPLE: self._calculate_pattern_score(full_text, self.simple_patterns),
            TaskComplexity.MODERATE: self._calculate_pattern_score(full_text, self.moderate_patterns),
            TaskComplexity.COMPLEX: self._calculate_pattern_score(full_text, self.complex_patterns),
            TaskComplexity.CRITICAL: self._calculate_pattern_score(full_text, self.critical_patterns)
        }
        
        # Add keyword scoring
        for complexity, keywords in self.complexity_keywords.items():
            keyword_score = sum(1 for keyword in keywords if keyword in full_text)
            complexity_scores[complexity] += keyword_score * 0.1
        
        # Context-based adjustments
        if agent_type in ['trader', 'risk_manager', 'research_manager']:
            complexity_scores[TaskComplexity.COMPLEX] += 0.3
            complexity_scores[TaskComplexity.CRITICAL] += 0.5
        
        if task_stage in ['final_decision', 'investment_plan', 'risk_assessment']:
            complexity_scores[TaskComplexity.CRITICAL] += 0.4
        
        # Message length consideration
        text_length = len(full_text)
        if text_length > 1000:
            complexity_scores[TaskComplexity.COMPLEX] += 0.2
        elif text_length > 2000:
            complexity_scores[TaskComplexity.CRITICAL] += 0.3
        
        # Determine final complexity
        max_complexity = max(complexity_scores, key=complexity_scores.get)
        confidence = complexity_scores[max_complexity]
        
        # Determine recommended model
        if max_complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
            if confidence < 0.7 and text_length > 500:
                # Borderline case - use Sonnet for safety
                recommended_model = ModelType.SONNET
                reasoning = f"Borderline {max_complexity.value} task, using Sonnet for safety"
            else:
                recommended_model = ModelType.HAIKU
                reasoning = f"Classified as {max_complexity.value}, suitable for Haiku"
        else:
            recommended_model = ModelType.SONNET
            reasoning = f"Classified as {max_complexity.value}, requires Sonnet's advanced reasoning"
        
        # Estimate token usage
        estimated_tokens = max(100, len(full_text.split()) * 1.3)  # Rough estimate
        
        return TaskClassification(
            complexity=max_complexity,
            confidence=confidence,
            reasoning=reasoning,
            recommended_model=recommended_model,
            estimated_tokens=int(estimated_tokens)
        )
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matching"""
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.2
        return score


class IntelligentModelRouter:
    """
    Intelligent router that selects the optimal AI model for each task
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model router
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize models
        self.models = {
            ModelType.HAIKU: DirectChatAnthropic(model=ModelType.HAIKU.value),
            ModelType.SONNET: DirectChatAnthropic(model=ModelType.SONNET.value)
        }
        
        # Performance tracking
        self.metrics = {
            ModelType.HAIKU: ModelMetrics("claude-3.5-haiku"),
            ModelType.SONNET: ModelMetrics("claude-3.5-sonnet")
        }
        
        # Task analyzer
        self.analyzer = TaskComplexityAnalyzer()
        
        # Router settings
        self.force_model = self.config.get('force_model')  # Override for testing
        self.cost_optimization = self.config.get('cost_optimization', True)
        self.performance_optimization = self.config.get('performance_optimization', True)
        
        logger.info("🎯 Intelligent Model Router initialized")
    
    def route_request(self, messages: List[BaseMessage], context: Dict[str, Any] = None) -> Tuple[Any, TaskClassification]:
        """
        Route request to the optimal model
        
        Args:
            messages: Messages for the AI model
            context: Additional context for routing decision
            
        Returns:
            Tuple of (model_response, task_classification)
        """
        start_time = time.time()
        
        # Override if force_model is set
        if self.force_model:
            model_type = ModelType(self.force_model)
            classification = TaskClassification(
                complexity=TaskComplexity.COMPLEX,
                confidence=1.0,
                reasoning="Forced model selection",
                recommended_model=model_type,
                estimated_tokens=500
            )
        else:
            # Classify task complexity
            classification = self.analyzer.classify_task(messages, context)
        
        # Select model
        selected_model = classification.recommended_model
        model = self.models[selected_model]
        
        logger.info(f"🎯 Routing to {selected_model.name}: {classification.reasoning}")
        
        try:
            # Execute the request
            response = model.invoke(messages)
            
            # Update metrics
            latency = time.time() - start_time
            self._update_metrics(selected_model, latency, classification.estimated_tokens, True)
            
            logger.info(f"✅ {selected_model.name} completed in {latency:.2f}s")
            
            return response, classification
            
        except Exception as e:
            logger.error(f"❌ {selected_model.name} failed: {e}")
            
            # Update metrics for failure
            latency = time.time() - start_time
            self._update_metrics(selected_model, latency, classification.estimated_tokens, False)
            
            # Fallback strategy
            if selected_model == ModelType.HAIKU:
                logger.info("🔄 Falling back to Sonnet after Haiku failure")
                try:
                    fallback_model = self.models[ModelType.SONNET]
                    response = fallback_model.invoke(messages)
                    
                    fallback_latency = time.time() - start_time
                    self._update_metrics(ModelType.SONNET, fallback_latency, classification.estimated_tokens, True)
                    
                    return response, classification
                except Exception as fallback_e:
                    logger.error(f"❌ Fallback to Sonnet also failed: {fallback_e}")
                    raise fallback_e
            else:
                raise e
    
    def _update_metrics(self, model_type: ModelType, latency: float, estimated_tokens: int, success: bool):
        """Update performance metrics for a model"""
        metrics = self.metrics[model_type]
        
        metrics.total_requests += 1
        metrics.total_tokens += estimated_tokens
        metrics.last_used = datetime.now()
        
        # Update latency (moving average)
        if metrics.average_latency == 0:
            metrics.average_latency = latency
        else:
            metrics.average_latency = (metrics.average_latency * 0.9) + (latency * 0.1)
        
        # Update success rate (moving average)
        if success:
            metrics.success_rate = (metrics.success_rate * 0.95) + (1.0 * 0.05)
        else:
            metrics.success_rate = (metrics.success_rate * 0.95) + (0.0 * 0.05)
        
        # Estimate cost (rough approximation)
        if model_type == ModelType.HAIKU:
            cost_per_token = 0.00025 / 1000  # Approximate
        else:
            cost_per_token = 0.003 / 1000    # Approximate
        
        metrics.total_cost += estimated_tokens * cost_per_token
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {
            "routing_stats": {},
            "cost_analysis": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_cost = sum(m.total_cost for m in self.metrics.values())
        
        for model_type, metrics in self.metrics.items():
            if metrics.total_requests > 0:
                summary["routing_stats"][model_type.name] = {
                    "requests": metrics.total_requests,
                    "percentage": (metrics.total_requests / total_requests) * 100,
                    "average_latency": metrics.average_latency,
                    "success_rate": metrics.success_rate,
                    "total_tokens": metrics.total_tokens,
                    "total_cost": metrics.total_cost,
                    "cost_per_request": metrics.total_cost / metrics.total_requests,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else None
                }
        
        # Cost analysis
        haiku_metrics = self.metrics[ModelType.HAIKU]
        sonnet_metrics = self.metrics[ModelType.SONNET]
        
        if haiku_metrics.total_requests > 0 and sonnet_metrics.total_requests > 0:
            cost_savings = (sonnet_metrics.total_cost - haiku_metrics.total_cost)
            summary["cost_analysis"] = {
                "total_cost": total_cost,
                "haiku_cost": haiku_metrics.total_cost,
                "sonnet_cost": sonnet_metrics.total_cost,
                "estimated_savings": cost_savings,
                "savings_percentage": (cost_savings / (haiku_metrics.total_cost + sonnet_metrics.total_cost)) * 100
            }
        
        # Performance analysis
        summary["performance_analysis"] = {
            "total_requests": total_requests,
            "average_latency": sum(m.average_latency * m.total_requests for m in self.metrics.values()) / max(1, total_requests),
            "overall_success_rate": sum(m.success_rate * m.total_requests for m in self.metrics.values()) / max(1, total_requests)
        }
        
        # Recommendations
        if haiku_metrics.success_rate < 0.9:
            summary["recommendations"].append("Consider increasing Haiku usage threshold - lower success rate detected")
        
        if sonnet_metrics.total_requests < total_requests * 0.3:
            summary["recommendations"].append("Sonnet usage is low - consider reviewing task classification")
        
        if total_cost > 50.0:  # Arbitrary threshold
            summary["recommendations"].append("High AI costs detected - review model selection strategy")
        
        return summary
    
    def optimize_routing_strategy(self) -> Dict[str, Any]:
        """Analyze and optimize the routing strategy"""
        summary = self.get_performance_summary()
        
        # Analyze patterns and suggest optimizations
        optimizations = {
            "current_performance": summary,
            "suggested_changes": [],
            "estimated_impact": {}
        }
        
        haiku_metrics = self.metrics[ModelType.HAIKU]
        sonnet_metrics = self.metrics[ModelType.SONNET]
        
        # Cost optimization suggestions
        if sonnet_metrics.total_requests > haiku_metrics.total_requests * 2:
            optimizations["suggested_changes"].append({
                "type": "cost_optimization",
                "description": "Increase Haiku usage for simple tasks",
                "estimated_savings": "15-25%"
            })
        
        # Performance optimization suggestions
        if haiku_metrics.average_latency > sonnet_metrics.average_latency:
            optimizations["suggested_changes"].append({
                "type": "performance_investigation",
                "description": "Investigate Haiku latency issues",
                "estimated_improvement": "10-20% faster response"
            })
        
        return optimizations


def create_intelligent_router(config: Dict[str, Any] = None) -> IntelligentModelRouter:
    """
    Factory function to create an intelligent model router
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured IntelligentModelRouter instance
    """
    return IntelligentModelRouter(config)


# Global router instance for singleton usage
_global_router = None

def get_global_router(config: Dict[str, Any] = None) -> IntelligentModelRouter:
    """Get or create the global model router"""
    global _global_router
    if _global_router is None:
        _global_router = IntelligentModelRouter(config)
    return _global_router


if __name__ == "__main__":
    # Test the intelligent model router
    router = IntelligentModelRouter()
    
    # Test simple task
    simple_messages = [HumanMessage(content="What is the current price of AAPL?")]
    
    try:
        response, classification = router.route_request(simple_messages)
        print(f"Simple task: {classification.complexity.value} -> {classification.recommended_model.name}")
    except Exception as e:
        print(f"Simple task failed: {e}")
    
    # Test complex task
    complex_messages = [HumanMessage(content="Provide a comprehensive investment analysis and recommendation for TSLA including technical analysis, fundamental analysis, risk assessment, and final buy/sell/hold decision with detailed reasoning.")]
    
    try:
        response, classification = router.route_request(complex_messages)
        print(f"Complex task: {classification.complexity.value} -> {classification.recommended_model.name}")
    except Exception as e:
        print(f"Complex task failed: {e}")
    
    # Get performance summary
    summary = router.get_performance_summary()
    print(f"Performance summary: {summary}") 