"""
Dynamic Agent Orchestration System.

This system enables the planner to dynamically recruit and dismiss agents
based on task requirements, complexity, and resource availability.

Key Features:
- Task-based agent recruitment
- Dynamic agent pool management
- Resource-aware scheduling
- Agent performance tracking
- Automatic agent dismissal
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from ..agents.base import Agent, AgentResponse
from ..agents.planner import Plan, PlannerAgent, TaskComplexity, TaskType
from ..config.settings import AgentConfig, ModelEndpoint
from ..observability.logging import get_logger
from ..observability.tracing import trace_span

logger = get_logger(__name__)


class AgentRole(Enum):
    """Available agent roles for recruitment."""
    
    PLANNER = "planner"
    CODER = "coder"
    CRITIC = "critic"
    RESEARCHER = "researcher"
    TESTER = "tester"
    OPTIMIZER = "optimizer"
    DOCUMENTER = "documenter"
    DEBUGGER = "debugger"


class AgentStatus(Enum):
    """Agent status in the orchestration system."""
    
    AVAILABLE = "available"
    BUSY = "busy"
    ACTIVE = "active"  # Backward-compat alias used by tests
    IDLE = "idle"
    DISMISSED = "dismissed"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    avg_confidence: float = 0.0
    last_used: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_tasks / max(1, self.total_tasks)
    
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score."""
        base_score = self.success_rate * 0.6 + self.avg_confidence * 0.4
        
        # Penalize slow agents
        if self.avg_execution_time > 30.0:
            base_score *= 0.8
        elif self.avg_execution_time > 60.0:
            base_score *= 0.6
            
        return min(1.0, base_score)


@dataclass
class ManagedAgent:
    """Wrapper for agents in the orchestration system."""
    
    agent: Agent
    role: AgentRole
    status: AgentStatus = AgentStatus.AVAILABLE
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    created_at: float = field(default_factory=time.time)
    current_task: Optional[str] = None
    task_id: Optional[str] = None  # Backward-compat field name expected by tests


class TaskRequirement:
    """Requirements for task execution."""
    
    def __init__(
        self,
        required_roles: List[AgentRole],
        optional_roles: Optional[List[AgentRole]] = None,
        max_agents: int = 5,
        min_performance_score: float = 0.6,
        task_type: Optional[TaskType] = None,
        complexity: Optional[TaskComplexity] = None,
        estimated_complexity: float | None = None,
        estimated_duration: int | None = None,
        resource_requirements: Optional[Dict[str, Any]] = None,
    ):
        self.required_roles = required_roles
        self.optional_roles = optional_roles or []
        self.max_agents = max_agents
        self.min_performance_score = min_performance_score
        self.task_type = task_type
        self.complexity = complexity
        self.estimated_complexity = estimated_complexity if estimated_complexity is not None else 0.0
        self.estimated_duration = estimated_duration if estimated_duration is not None else 0
        self.resource_requirements = resource_requirements or {}


class DynamicOrchestrator:
    """
    Dynamic agent orchestration system.
    
    Features:
    - Automatic agent recruitment based on task requirements
    - Performance-based agent selection
    - Resource-aware scheduling
    - Automatic agent dismissal for idle agents
    - Agent pool optimization
    """
    
    def __init__(
        self,
        max_agents: int = 10,
        idle_timeout: float = 300.0,  # 5 minutes
        performance_threshold: float = 0.5,
    ):
        self.max_agents = max_agents
        self.idle_timeout = idle_timeout
        self.performance_threshold = performance_threshold
        
        # Agent management
        self.agents: Dict[str, ManagedAgent] = {}
        self.agent_factories: Dict[AgentRole, Type[Agent]] = {}
        self.role_assignments: Dict[AgentRole, List[str]] = defaultdict(list)
        
        # Task management
        self.active_tasks: Dict[str, TaskRequirement] = {}
        self.task_assignments: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.recruitment_history: List[Dict[str, Any]] = []
        self.dismissal_history: List[Dict[str, Any]] = []
        
        # Initialize with planner
        self._initialize_core_agents()
    
    def _initialize_core_agents(self) -> None:
        """Initialize core agents that are always available."""
        # Register agent factories (would be populated by dependency injection)
        from ..agents.planner import PlannerAgent
        self.agent_factories[AgentRole.PLANNER] = PlannerAgent
        # Do not auto-create a planner here; tests expect zero agents at start
    
    def register_agent_factory(self, role: AgentRole, factory: Type[Agent]) -> None:
        """Register an agent factory for a specific role."""
        self.agent_factories[role] = factory
        logger.info(f"Registered agent factory for role: {role.value}")
    
    @trace_span("orchestrator.analyze_task_requirements")
    async def analyze_task_requirements(self, query: str, context: Optional[Dict[str, Any]] = None) -> TaskRequirement:
        """Analyze a task to determine agent requirements."""
        # Heuristic analysis based on query keywords (fast path, no model required)
        q = (query or "").lower()
        required_roles: List[AgentRole] = [AgentRole.PLANNER]
        optional_roles: List[AgentRole] = []
        task_type: TaskType | None = None
        complexity: TaskComplexity | None = None
        est_complexity = 0.4
        est_duration = 60
        
        if any(w in q for w in ["implement", "code", "build", "develop", "create", "add endpoint", "write"]):
            if AgentRole.CODER not in required_roles:
                required_roles.append(AgentRole.CODER)
            task_type = TaskType.IMPLEMENTATION
            est_complexity = 0.7
            est_duration = 300
        
        if any(w in q for w in ["review", "critique", "check", "validate", "audit"]):
            if AgentRole.CRITIC not in required_roles:
                required_roles.append(AgentRole.CRITIC)
            task_type = task_type or TaskType.ANALYSIS
            est_complexity = max(est_complexity, 0.5)
            est_duration = max(est_duration, 180)
        
        if any(w in q for w in ["research", "investigate", "analyze", "study"]):
            optional_roles.append(AgentRole.RESEARCHER)
            task_type = task_type or TaskType.RESEARCH
        
        # Try to refine with planner if available (best-effort)
        try:
            planner_id = self._get_agent_by_role(AgentRole.PLANNER) or self._create_agent(AgentRole.PLANNER)
            if planner_id:
                planner = self.agents[planner_id]
                analysis_query = f"Analyze this task and suggest roles and complexity: {query}\nContext: {context or {}}"
                response = await planner.agent.process(analysis_query, context)
                plan = await self._extract_plan_from_response(response)
                if plan:
                    return self._plan_to_requirements(plan, response.response)
        except Exception as e:
            logger.error(f"Failed to analyze task requirements: {e}")
        
        return TaskRequirement(
            required_roles=required_roles,
            optional_roles=optional_roles,
            max_agents=3,
            task_type=task_type,
            complexity=complexity,
            estimated_complexity=est_complexity,
            estimated_duration=est_duration,
        )
    
    @trace_span("orchestrator.recruit_agents")
    async def recruit_agents(self, task_id: str, requirements: TaskRequirement) -> List[str]:
        """Recruit agents for a specific task."""
        recruited_agents = []
        
        logger.info(f"Recruiting agents for task {task_id}: {requirements.required_roles}")
        
        # Deduplicate roles while preserving order
        seen: Set[AgentRole] = set()
        required_roles = [r for r in requirements.required_roles if not (r in seen or seen.add(r))]
        
        # First, try to assign existing agents
        for role in required_roles:
            agent_id = self._get_best_available_agent(role, requirements.min_performance_score)
            if agent_id:
                recruited_agents.append(agent_id)
                self._assign_agent_to_task(agent_id, task_id)
            else:
                # Create new agent if needed and possible
                if len(self.agents) < self.max_agents:
                    new_agent_id = self._create_agent(role)
                    if new_agent_id:
                        recruited_agents.append(new_agent_id)
                        self._assign_agent_to_task(new_agent_id, task_id)
                        
                        # Log recruitment
                        self.recruitment_history.append({
                            "timestamp": time.time(),
                            "task_id": task_id,
                            "role": role.value,
                            "agent_id": new_agent_id,
                            "reason": "required_role_not_available"
                        })
        
        # Try to recruit optional agents if we have capacity
        for role in (requirements.optional_roles or []):
            if len(recruited_agents) >= requirements.max_agents:
                break
                
            agent_id = self._get_best_available_agent(role, requirements.min_performance_score)
            if agent_id and agent_id not in recruited_agents:
                recruited_agents.append(agent_id)
                self._assign_agent_to_task(agent_id, task_id)
            elif len(self.agents) < self.max_agents:
                new_agent_id = self._create_agent(role)
                if new_agent_id:
                    recruited_agents.append(new_agent_id)
                    self._assign_agent_to_task(new_agent_id, task_id)
                    
                    self.recruitment_history.append({
                        "timestamp": time.time(),
                        "task_id": task_id,
                        "role": role.value,
                        "agent_id": new_agent_id,
                        "reason": "optional_role_enhancement"
                    })
        
        # Store task requirements
        self.active_tasks[task_id] = requirements
        
        logger.info(f"Recruited {len(recruited_agents)} agents for task {task_id}: {recruited_agents}")
        return recruited_agents
    
    @trace_span("orchestrator.dismiss_agents")
    async def dismiss_agents(self, task_id: str, agent_ids: Optional[List[str]] = None) -> int:
        """Dismiss agents after task completion or timeout."""
        dismissed_count = 0
        
        if agent_ids is None:
            # Dismiss all agents assigned to this task
            agent_ids = self.task_assignments.get(task_id, [])
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Update agent status
                agent.status = AgentStatus.AVAILABLE
                agent.current_task = None
                
                # Decide whether to keep or dismiss the agent
                should_dismiss = self._should_dismiss_agent(agent)
                
                if should_dismiss:
                    self._dismiss_agent(agent_id)
                    dismissed_count += 1
                    
                    # Log dismissal
                    self.dismissal_history.append({
                        "timestamp": time.time(),
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "role": agent.role.value,
                        "reason": "task_completion_dismissal",
                        "performance_score": agent.metrics.performance_score
                    })
        
        # Clean up task assignments
        if task_id in self.task_assignments:
            del self.task_assignments[task_id]
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        logger.info(f"Dismissed {dismissed_count} agents after task {task_id} completion")
        return dismissed_count
    
    async def cleanup_idle_agents(self) -> int:
        """Clean up idle agents that exceed the timeout."""
        dismissed_count = 0
        current_time = time.time()
        
        idle_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if (agent.status == AgentStatus.IDLE and 
                current_time - agent.metrics.last_used > self.idle_timeout)
        ]
        
        for agent_id in idle_agents:
            if agent_id in self.agents:
                role_val = self.agents[agent_id].role.value
                idle_time = current_time - self.agents[agent_id].metrics.last_used
            else:
                role_val = "unknown"
                idle_time = 0.0
            self._dismiss_agent(agent_id)
            dismissed_count += 1
            
            self.dismissal_history.append({
                "timestamp": current_time,
                "agent_id": agent_id,
                "role": role_val,
                "reason": "idle_timeout",
                "idle_time": idle_time,
            })
        
        if dismissed_count > 0:
            logger.info(f"Dismissed {dismissed_count} idle agents")
        
        return dismissed_count
    
    def _get_agent_by_role(self, role: AgentRole) -> Optional[str]:
        """Get first available agent with specified role."""
        for agent_id in self.role_assignments[role]:
            if agent_id in self.agents and self.agents[agent_id].status == AgentStatus.AVAILABLE:
                return agent_id
        return None
    
    def _get_best_available_agent(self, role: AgentRole, min_performance: float) -> Optional[str]:
        """Get the best available agent for a role."""
        candidates = [
            (agent_id, self.agents[agent_id]) 
            for agent_id in self.role_assignments[role]
            if (agent_id in self.agents and 
                self.agents[agent_id].status == AgentStatus.AVAILABLE and
                self.agents[agent_id].metrics.performance_score >= min_performance)
        ]
        
        if not candidates:
            return None
        
        # Sort by performance score (descending)
        candidates.sort(key=lambda x: x[1].metrics.performance_score, reverse=True)
        return candidates[0][0]
    
    def _create_agent(self, role: AgentRole) -> Optional[str]:
        """Create a new agent with the specified role."""
        if role not in self.agent_factories:
            logger.warning(f"No factory registered for role: {role.value}")
            return None
        
        try:
            agent_factory = self.agent_factories[role]
            # Support either a zero-arg callable or a class requiring defaults
            try:
                agent = agent_factory()
            except TypeError:
                # Instantiate with default config for tests
                default_endpoint = ModelEndpoint(name=f"mock-{role.value}", base_url="local")
                default_config = AgentConfig()
                agent = agent_factory(endpoint=default_endpoint, config=default_config)
            
            # Generate unique ID
            agent_id = f"{role.value}_{len(self.agents)}_{int(time.time())}"
            
            # Create managed agent
            managed_agent = ManagedAgent(
                agent=agent,
                role=role,
                status=AgentStatus.AVAILABLE
            )
            
            # Register agent
            self.agents[agent_id] = managed_agent
            self.role_assignments[role].append(agent_id)
            
            logger.info(f"Created new agent: {agent_id} ({role.value})")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent for role {role.value}: {e}")
            return None
    
    def _assign_agent_to_task(self, agent_id: str, task_id: str) -> None:
        """Assign an agent to a task."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = AgentStatus.ACTIVE
            agent.current_task = task_id
            agent.task_id = task_id
            agent.metrics.last_used = time.time()
            
            self.task_assignments[task_id].append(agent_id)
    
    def _should_dismiss_agent(self, agent: ManagedAgent) -> bool:
        """Determine if an agent should be dismissed."""
        # Dismiss if performance is too low
        if agent.metrics.performance_score < self.performance_threshold:
            return True
        
        # Dismiss if agent has been idle for too long
        if agent.status == AgentStatus.IDLE and agent.metrics.last_used < time.time() - self.idle_timeout:
            return True
        
        # Keep agent by default
        return False
    
    def _dismiss_agent(self, agent_id: str) -> None:
        """Dismiss an agent from the system."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = AgentStatus.DISMISSED
            
            # Remove from role assignments
            if agent_id in self.role_assignments[agent.role]:
                self.role_assignments[agent.role].remove(agent_id)
            
            # Remove from agents dict
            del self.agents[agent_id]
            
            logger.info(f"Dismissed agent: {agent_id} ({agent.role.value})")
    
    async def _extract_plan_from_response(self, response: AgentResponse) -> Optional[Plan]:
        """Extract plan from planner response."""
        # This would use the planner's extract_plan_structure method
        # For now, return None as fallback
        return None
    
    def _plan_to_requirements(self, plan: Optional[Plan], response_text: str) -> TaskRequirement:
        """Convert a plan to task requirements."""
        required_roles = [AgentRole.PLANNER]
        optional_roles = []
        
        # Analyze response text for role hints
        response_lower = response_text.lower()
        
        if any(word in response_lower for word in ["code", "implement", "build", "develop", "create", "add endpoint", "write"]):
            required_roles.append(AgentRole.CODER)
        
        if any(word in response_lower for word in ["review", "check", "validate", "critique"]):
            optional_roles.append(AgentRole.CRITIC)
        
        if any(word in response_lower for word in ["research", "investigate", "study", "analyze"]):
            optional_roles.append(AgentRole.RESEARCHER)
        
        if any(word in response_lower for word in ["test", "verify", "validate"]):
            optional_roles.append(AgentRole.TESTER)
        
        if any(word in response_lower for word in ["optimize", "improve", "performance"]):
            optional_roles.append(AgentRole.OPTIMIZER)
        
        if any(word in response_lower for word in ["document", "explain", "describe"]):
            optional_roles.append(AgentRole.DOCUMENTER)
        
        if any(word in response_lower for word in ["debug", "fix", "error", "bug"]):
            required_roles.append(AgentRole.DEBUGGER)
        
        # Determine complexity-based limits
        max_agents = 3
        if plan:
            if plan.complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
                max_agents = 5
            elif plan.complexity == TaskComplexity.MEDIUM:
                max_agents = 4
        
        return TaskRequirement(
            required_roles=required_roles,
            optional_roles=optional_roles,
            max_agents=max_agents,
            task_type=plan.task_type if plan else None,
            complexity=plan.complexity if plan else None
        )
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration system statistics."""
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a.status != AgentStatus.DISMISSED])
        idle_agents = len([a for a in self.agents.values() if a.status == AgentStatus.IDLE])
        role_distribution = {role.value: len(self.role_assignments[role]) for role in AgentRole}
        avg_performance = sum(a.metrics.performance_score for a in self.agents.values()) / max(1, len(self.agents))
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "idle_agents": idle_agents,
            "active_tasks": len(self.active_tasks),
            "role_distribution": role_distribution,
            "total_recruitments": len(self.recruitment_history),
            "total_dismissals": len(self.dismissal_history),
            "average_task_duration": 0.0,
            "avg_performance": avg_performance,
        }
