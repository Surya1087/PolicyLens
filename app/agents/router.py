from __future__ import annotations

from rag.models import PolicyLensError

from .claim_agent import ClaimAgent
from .comparison_agent import ComparisonAgent
from .insurance_agent import InsuranceAgent
from .risk_agent import RiskAgent
from .summary_agent import SummaryAgent


class Router:
    ROUTES = {
        "Ask Question": InsuranceAgent,
        "Executive Summary": SummaryAgent,
        "Risk Analysis": RiskAgent,
        "Claim Check": ClaimAgent,
        "Compare Policies": ComparisonAgent,
    }

    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm

    def run(self, route: str, **kwargs) -> dict:
        agent_type = self.ROUTES.get(route)
        if agent_type is None:
            raise PolicyLensError(f"Unknown route: {route}. Choose one of: {', '.join(self.ROUTES)}")
        return agent_type(self.pipeline, self.llm).run(**kwargs)


AgentRouter = Router
