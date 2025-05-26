"""
Universal Project Specification Schema

This module defines the schema for comprehensive project specifications
that can be used for any type of software project across any programming
language, framework, or domain.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ProjectType(str, Enum):
    """Common project types - extensible list"""
    CLI_TOOL = "cli_tool"
    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    LIBRARY = "library"
    API = "api"
    MICROSERVICE = "microservice"
    GAME = "game"
    DATA_PIPELINE = "data_pipeline"
    ML_MODEL = "ml_model"
    AUTOMATION_SCRIPT = "automation_script"
    PLUGIN = "plugin"
    EXTENSION = "extension"
    OTHER = "other"


class InterfaceType(str, Enum):
    """Interface types for different project categories"""
    COMMAND_LINE = "command_line"
    WEB_UI = "web_ui"
    MOBILE_UI = "mobile_ui"
    DESKTOP_GUI = "desktop_gui"
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    LIBRARY_API = "library_api"
    NO_INTERFACE = "no_interface"
    OTHER = "other"


class ComplexityLevel(str, Enum):
    """Project complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class TimelineType(str, Enum):
    """Development timeline expectations"""
    QUICK_PROTOTYPE = "quick_prototype"
    MVP = "mvp"
    PRODUCTION_READY = "production_ready"
    ENTERPRISE_GRADE = "enterprise_grade"


class TechnicalRequirements(BaseModel):
    """Technical specifications and constraints"""
    primary_language: str = Field(default="", description="Main programming language")
    secondary_languages: List[str] = Field(default_factory=list, description="Additional languages if polyglot")
    target_platforms: List[str] = Field(default_factory=list, description="Target platforms/operating systems")
    dependencies: Dict[str, List[str]] = Field(
        default_factory=lambda: {"required": [], "optional": []},
        description="Project dependencies"
    )
    performance_requirements: List[str] = Field(default_factory=list, description="Performance constraints and goals")
    security_requirements: List[str] = Field(default_factory=list, description="Security considerations")
    scalability_requirements: List[str] = Field(default_factory=list, description="Scalability needs")


class ProjectRequirements(BaseModel):
    """Functional and non-functional requirements"""
    functional: List[str] = Field(default_factory=list, description="What the software should do")
    non_functional: List[str] = Field(default_factory=list, description="How the software should perform")
    constraints: List[str] = Field(default_factory=list, description="Limitations or restrictions")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made during planning")


class InterfaceSpecification(BaseModel):
    """Interface design and interaction specifications"""
    type: InterfaceType = Field(default=InterfaceType.OTHER, description="Type of user interface")
    style: str = Field(default="", description="UI/UX style preferences")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Interface-specific details")
    accessibility_requirements: List[str] = Field(default_factory=list, description="Accessibility considerations")


class ArchitectureSpecification(BaseModel):
    """Software architecture and design specifications"""
    patterns: List[str] = Field(default_factory=list, description="Design patterns to implement")
    modules: List[str] = Field(default_factory=list, description="High-level module structure")
    data_storage: str = Field(default="", description="Data storage approach")
    testing_strategy: str = Field(default="", description="Testing approach and frameworks")
    documentation_strategy: str = Field(default="", description="Documentation approach")
    code_style: str = Field(default="", description="Code style and formatting preferences")


class DeploymentSpecification(BaseModel):
    """Deployment and distribution specifications"""
    packaging: str = Field(default="", description="How the software will be packaged")
    distribution: str = Field(default="", description="How the software will be distributed")
    documentation: List[str] = Field(default_factory=list, description="Required documentation types")
    ci_cd_requirements: List[str] = Field(default_factory=list, description="CI/CD pipeline needs")
    monitoring_requirements: List[str] = Field(default_factory=list, description="Monitoring and observability")


class ProjectContext(BaseModel):
    """Additional context about the project"""
    domain: str = Field(default="", description="Problem domain or industry")
    complexity: ComplexityLevel = Field(default=ComplexityLevel.MEDIUM, description="Overall project complexity")
    timeline: TimelineType = Field(default=TimelineType.PRODUCTION_READY, description="Development timeline expectation")
    team_size: str = Field(default="", description="Expected team size")
    budget_constraints: List[str] = Field(default_factory=list, description="Budget or resource constraints")
    compliance_requirements: List[str] = Field(default_factory=list, description="Regulatory or compliance needs")


class ProjectSpecification(BaseModel):
    """Universal project specification that works for any software project"""
    
    # Basic project information
    name: str = Field(default="", description="Project name")
    type: ProjectType = Field(default=ProjectType.OTHER, description="Type of project")
    description: str = Field(default="", description="Brief project description")
    target_audience: str = Field(default="", description="Who will use this software")
    
    # Technical specifications
    technical: TechnicalRequirements = Field(default_factory=TechnicalRequirements)
    
    # Requirements
    requirements: ProjectRequirements = Field(default_factory=ProjectRequirements)
    
    # Interface design
    interface: InterfaceSpecification = Field(default_factory=InterfaceSpecification)
    
    # Architecture
    architecture: ArchitectureSpecification = Field(default_factory=ArchitectureSpecification)
    
    # Deployment
    deployment: DeploymentSpecification = Field(default_factory=DeploymentSpecification)
    
    # Context
    context: ProjectContext = Field(default_factory=ProjectContext)
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-friendly dictionary format"""
        def get_enum_value(field):
            """Safely get enum value, handling both enum objects and strings"""
            if hasattr(field, 'value'):
                return field.value
            elif isinstance(field, str):
                return field
            else:
                return ""
        
        return {
            "project": {
                "name": self.name,
                "type": get_enum_value(self.type),
                "description": self.description,
                "target_audience": self.target_audience
            },
            "technical": {
                "primary_language": self.technical.primary_language,
                "secondary_languages": self.technical.secondary_languages,
                "target_platforms": self.technical.target_platforms,
                "dependencies": self.technical.dependencies,
                "performance_requirements": self.technical.performance_requirements,
                "security_requirements": self.technical.security_requirements,
                "scalability_requirements": self.technical.scalability_requirements
            },
            "requirements": {
                "functional": self.requirements.functional,
                "non_functional": self.requirements.non_functional,
                "constraints": self.requirements.constraints,
                "assumptions": self.requirements.assumptions
            },
            "interface": {
                "type": get_enum_value(self.interface.type),
                "style": self.interface.style,
                "specifications": self.interface.specifications,
                "accessibility_requirements": self.interface.accessibility_requirements
            },
            "architecture": {
                "patterns": self.architecture.patterns,
                "modules": self.architecture.modules,
                "data_storage": self.architecture.data_storage,
                "testing_strategy": self.architecture.testing_strategy,
                "documentation_strategy": self.architecture.documentation_strategy,
                "code_style": self.architecture.code_style
            },
            "deployment": {
                "packaging": self.deployment.packaging,
                "distribution": self.deployment.distribution,
                "documentation": self.deployment.documentation,
                "ci_cd_requirements": self.deployment.ci_cd_requirements,
                "monitoring_requirements": self.deployment.monitoring_requirements
            },
            "context": {
                "domain": self.context.domain,
                "complexity": get_enum_value(self.context.complexity),
                "timeline": get_enum_value(self.context.timeline),
                "team_size": self.context.team_size,
                "budget_constraints": self.context.budget_constraints,
                "compliance_requirements": self.context.compliance_requirements
            }
        }

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "ProjectSpecification":
        """Create ProjectSpecification from YAML dictionary"""
        return cls(
            name=data.get("project", {}).get("name", ""),
            type=ProjectType(data.get("project", {}).get("type", "other")),
            description=data.get("project", {}).get("description", ""),
            target_audience=data.get("project", {}).get("target_audience", ""),
            technical=TechnicalRequirements(**data.get("technical", {})),
            requirements=ProjectRequirements(**data.get("requirements", {})),
            interface=InterfaceSpecification(
                type=InterfaceType(data.get("interface", {}).get("type", "other")),
                style=data.get("interface", {}).get("style", ""),
                specifications=data.get("interface", {}).get("specifications", {}),
                accessibility_requirements=data.get("interface", {}).get("accessibility_requirements", [])
            ),
            architecture=ArchitectureSpecification(**data.get("architecture", {})),
            deployment=DeploymentSpecification(**data.get("deployment", {})),
            context=ProjectContext(
                domain=data.get("context", {}).get("domain", ""),
                complexity=ComplexityLevel(data.get("context", {}).get("complexity", "medium")),
                timeline=TimelineType(data.get("context", {}).get("timeline", "production_ready")),
                team_size=data.get("context", {}).get("team_size", ""),
                budget_constraints=data.get("context", {}).get("budget_constraints", []),
                compliance_requirements=data.get("context", {}).get("compliance_requirements", [])
            )
        )


# Template for empty project specification
EMPTY_PROJECT_SPECIFICATION_TEMPLATE = """# Universal Project Specification
# This file contains comprehensive requirements for your project
# Generated by Chungoid Interactive Requirements Agent

project:
  name: ""
  type: ""  # cli_tool, web_app, mobile_app, desktop_app, library, api, etc.
  description: ""
  target_audience: ""

technical:
  primary_language: ""  # python, javascript, rust, go, java, etc.
  secondary_languages: []  # Additional languages for polyglot projects
  target_platforms: []  # linux, windows, macos, web, ios, android, etc.
  dependencies:
    required: []
    optional: []
  performance_requirements: []
  security_requirements: []
  scalability_requirements: []

requirements:
  functional: []  # What the software should do
  non_functional: []  # How the software should perform
  constraints: []  # Limitations or restrictions
  assumptions: []  # Assumptions made during planning

interface:
  type: ""  # command_line, web_ui, mobile_ui, desktop_gui, rest_api, etc.
  style: ""  # professional, casual, minimal, rich, etc.
  specifications: {}  # Interface-specific details
  accessibility_requirements: []

architecture:
  patterns: []  # Design patterns to implement
  modules: []  # High-level module structure
  data_storage: ""  # file, database, memory, cloud, etc.
  testing_strategy: ""
  documentation_strategy: ""
  code_style: ""

deployment:
  packaging: ""  # pip, npm, docker, binary, app_store, etc.
  distribution: ""  # github, package_registry, app_store, etc.
  documentation: []  # README, API_docs, user_guide, etc.
  ci_cd_requirements: []
  monitoring_requirements: []

context:
  domain: ""  # web_dev, data_science, gaming, finance, etc.
  complexity: ""  # simple, medium, complex, enterprise
  timeline: ""  # quick_prototype, mvp, production_ready, enterprise_grade
  team_size: ""
  budget_constraints: []
  compliance_requirements: []
""" 