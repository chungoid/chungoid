"""Project Type Detection Service

This module provides autonomous project type detection capabilities for software projects,
enabling adaptive agent behavior based on detected project characteristics.

Key Features:
- Multi-signal project type detection (file patterns, configs, structure)
- Framework and tool detection (Flask, Express, Django, React, etc.)
- Confidence scoring for detection reliability
- Extensible detection rules for new project types
- Integration with Smart Dependency Analysis Service

Design Principles:
- Multi-signal analysis for robust detection
- Confidence-based decision making
- Extensible rule engine for new project types
- Fast execution with caching support

Architectural Note:
This service appropriately uses hardcoded detection rules because it analyzes PROJECT STRUCTURE
(file patterns, config files, directory layouts) rather than code semantics. This is different
from the Smart Dependency Analysis Service which analyzes import statements and should rely on
LLM semantic understanding. Structural analysis benefits from explicit, fast pattern matching.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from pydantic import BaseModel, Field

from .llm_provider import LLMProvider
from .exceptions import ChungoidError

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================

class ProjectTypeDetectionError(ChungoidError):
    """Raised when project type detection operations fail."""
    pass

class UnsupportedDetectionMethodError(ProjectTypeDetectionError):
    """Raised when using unsupported detection methods."""
    pass

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class DetectionRule:
    """A rule for detecting project characteristics."""
    name: str
    description: str
    weight: float  # Importance weight 0.0-1.0
    required_files: List[str] = field(default_factory=list)
    required_patterns: List[str] = field(default_factory=list)  # Regex patterns
    forbidden_files: List[str] = field(default_factory=list)
    min_file_count: Dict[str, int] = field(default_factory=dict)  # Extension -> min count
    directory_patterns: List[str] = field(default_factory=list)

class ProjectCharacteristic(BaseModel):
    """A detected characteristic of a project."""
    name: str = Field(..., description="Characteristic name")
    category: str = Field(..., description="Category (language, framework, tool, etc.)")
    confidence: float = Field(..., description="Detection confidence 0.0-1.0")
    evidence: List[str] = Field(default_factory=list, description="Evidence that supports this detection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ProjectTypeDetectionResult(BaseModel):
    """Complete result of project type detection."""
    primary_language: str = Field(..., description="Primary programming language")
    language_confidence: float = Field(..., description="Confidence in language detection")
    frameworks: List[ProjectCharacteristic] = Field(default_factory=list)
    build_tools: List[ProjectCharacteristic] = Field(default_factory=list)
    testing_frameworks: List[ProjectCharacteristic] = Field(default_factory=list)
    deployment_tools: List[ProjectCharacteristic] = Field(default_factory=list)
    project_structure_type: Optional[str] = Field(None, description="Detected project structure pattern")
    config_files: List[str] = Field(default_factory=list)
    characteristics: List[ProjectCharacteristic] = Field(default_factory=list)
    overall_confidence: float = Field(..., description="Overall detection confidence")
    detection_metadata: Dict[str, Any] = Field(default_factory=dict)

# ============================================================================
# Detection Rules
# ============================================================================

class ProjectTypeDetectionRules:
    """Centralized collection of detection rules for different project types."""
    
    @staticmethod
    def get_language_rules() -> Dict[str, DetectionRule]:
        """Get rules for detecting programming languages."""
        return {
            "python": DetectionRule(
                name="Python",
                description="Python programming language",
                weight=1.0,
                required_files=["*.py"],
                required_patterns=[r"#!/usr/bin/env python", r"from\s+\w+\s+import", r"import\s+\w+"],
                min_file_count={"py": 1}
            ),
            "javascript": DetectionRule(
                name="JavaScript",
                description="JavaScript programming language",
                weight=1.0,
                required_files=["*.js"],
                required_patterns=[r"require\s*\(", r"import\s+.*\s+from", r"module\.exports"],
                min_file_count={"js": 1}
            ),
            "typescript": DetectionRule(
                name="TypeScript",
                description="TypeScript programming language",
                weight=1.0,
                required_files=["*.ts", "tsconfig.json"],
                required_patterns=[r"interface\s+\w+", r"type\s+\w+\s*=", r"export\s+type"],
                min_file_count={"ts": 1}
            ),
            "java": DetectionRule(
                name="Java",
                description="Java programming language",
                weight=1.0,
                required_files=["*.java"],
                required_patterns=[r"public\s+class\s+\w+", r"package\s+[\w\.]+"],
                min_file_count={"java": 1}
            ),
            "rust": DetectionRule(
                name="Rust",
                description="Rust programming language",
                weight=1.0,
                required_files=["Cargo.toml", "*.rs"],
                required_patterns=[r"fn\s+main\s*\(", r"use\s+\w+"],
                min_file_count={"rs": 1}
            ),
            "go": DetectionRule(
                name="Go",
                description="Go programming language",
                weight=1.0,
                required_files=["go.mod", "*.go"],
                required_patterns=[r"package\s+main", r"func\s+\w+"],
                min_file_count={"go": 1}
            )
        }
    
    @staticmethod
    def get_framework_rules() -> Dict[str, DetectionRule]:
        """Get rules for detecting frameworks."""
        return {
            "flask": DetectionRule(
                name="Flask",
                description="Python Flask web framework",
                weight=0.9,
                required_patterns=[r"from\s+flask\s+import", r"Flask\s*\(", r"@app\.route"],
                required_files=["*.py"]
            ),
            "django": DetectionRule(
                name="Django",
                description="Python Django web framework",
                weight=0.9,
                required_files=["manage.py", "settings.py"],
                required_patterns=[r"from\s+django", r"DJANGO_SETTINGS_MODULE"],
                directory_patterns=["*/migrations/"]
            ),
            "fastapi": DetectionRule(
                name="FastAPI",
                description="Python FastAPI web framework",
                weight=0.9,
                required_patterns=[r"from\s+fastapi\s+import", r"FastAPI\s*\(", r"@app\.(get|post|put|delete)"],
                required_files=["*.py"]
            ),
            "express": DetectionRule(
                name="Express",
                description="Node.js Express web framework",
                weight=0.9,
                required_patterns=[r"require\s*\(\s*['\"]express['\"]", r"express\s*\(\s*\)", r"app\.(get|post|put|delete)"],
                required_files=["*.js", "*.ts"]
            ),
            "react": DetectionRule(
                name="React",
                description="React frontend framework",
                weight=0.9,
                required_patterns=[r"import\s+React", r"from\s+['\"]react['\"]", r"jsx|tsx"],
                required_files=["*.jsx", "*.tsx"],
                min_file_count={"jsx": 1, "tsx": 1}
            ),
            "vue": DetectionRule(
                name="Vue.js",
                description="Vue.js frontend framework",
                weight=0.9,
                required_patterns=[r"import\s+Vue", r"from\s+['\"]vue['\"]", r"\.vue\s*$"],
                required_files=["*.vue"],
                min_file_count={"vue": 1}
            ),
            "angular": DetectionRule(
                name="Angular",
                description="Angular frontend framework",
                weight=0.9,
                required_files=["angular.json", "*.component.ts"],
                required_patterns=[r"@Component", r"@Injectable", r"from\s+['\"]@angular"],
                directory_patterns=["src/app/"]
            ),
            "nextjs": DetectionRule(
                name="Next.js",
                description="Next.js React framework",
                weight=0.9,
                required_files=["next.config.js", "pages/*.js", "pages/*.tsx"],
                required_patterns=[r"from\s+['\"]next", r"export\s+default\s+function"],
                directory_patterns=["pages/", ".next/"]
            )
        }
    
    @staticmethod
    def get_build_tool_rules() -> Dict[str, DetectionRule]:
        """Get rules for detecting build tools."""
        return {
            "pip": DetectionRule(
                name="pip",
                description="Python pip package manager",
                weight=0.8,
                required_files=["requirements.txt", "requirements*.txt"]
            ),
            "poetry": DetectionRule(
                name="Poetry",
                description="Python Poetry dependency manager",
                weight=0.8,
                required_files=["pyproject.toml", "poetry.lock"]
            ),
            "pipenv": DetectionRule(
                name="Pipenv",
                description="Python Pipenv environment manager",
                weight=0.8,
                required_files=["Pipfile", "Pipfile.lock"]
            ),
            "conda": DetectionRule(
                name="Conda",
                description="Conda package manager",
                weight=0.8,
                required_files=["environment.yml", "conda.yml", "environment.yaml"]
            ),
            "npm": DetectionRule(
                name="npm",
                description="Node.js npm package manager",
                weight=0.8,
                required_files=["package.json", "package-lock.json"]
            ),
            "yarn": DetectionRule(
                name="Yarn",
                description="Node.js Yarn package manager",
                weight=0.8,
                required_files=["yarn.lock", "package.json"]
            ),
            "pnpm": DetectionRule(
                name="pnpm",
                description="Node.js pnpm package manager",
                weight=0.8,
                required_files=["pnpm-lock.yaml", "package.json"]
            ),
            "webpack": DetectionRule(
                name="Webpack",
                description="Webpack module bundler",
                weight=0.7,
                required_files=["webpack.config.js", "webpack.config.ts"]
            ),
            "vite": DetectionRule(
                name="Vite",
                description="Vite build tool",
                weight=0.7,
                required_files=["vite.config.js", "vite.config.ts"]
            ),
            "gradle": DetectionRule(
                name="Gradle",
                description="Gradle build tool",
                weight=0.8,
                required_files=["build.gradle", "build.gradle.kts", "gradlew"]
            ),
            "maven": DetectionRule(
                name="Maven",
                description="Maven build tool",
                weight=0.8,
                required_files=["pom.xml"]
            ),
            "cargo": DetectionRule(
                name="Cargo",
                description="Rust Cargo build tool",
                weight=0.8,
                required_files=["Cargo.toml", "Cargo.lock"]
            ),
            "makefile": DetectionRule(
                name="Make",
                description="GNU Make build tool",
                weight=0.6,
                required_files=["Makefile", "makefile", "GNUmakefile"]
            )
        }
    
    @staticmethod
    def get_testing_framework_rules() -> Dict[str, DetectionRule]:
        """Get rules for detecting testing frameworks."""
        return {
            "pytest": DetectionRule(
                name="pytest",
                description="Python pytest testing framework",
                weight=0.8,
                required_files=["pytest.ini", "test_*.py", "tests/*.py"],
                required_patterns=[r"import\s+pytest", r"def\s+test_\w+"]
            ),
            "unittest": DetectionRule(
                name="unittest",
                description="Python unittest framework",
                weight=0.7,
                required_patterns=[r"import\s+unittest", r"class\s+\w+Test\w*", r"def\s+test_\w+"]
            ),
            "jest": DetectionRule(
                name="Jest",
                description="JavaScript Jest testing framework",
                weight=0.8,
                required_files=["jest.config.js", "*.test.js", "*.spec.js"],
                required_patterns=[r"require\s*\(\s*['\"]jest['\"]", r"describe\s*\(", r"test\s*\(", r"it\s*\("]
            ),
            "mocha": DetectionRule(
                name="Mocha",
                description="JavaScript Mocha testing framework",
                weight=0.8,
                required_patterns=[r"require\s*\(\s*['\"]mocha['\"]", r"describe\s*\(", r"it\s*\("]
            ),
            "cypress": DetectionRule(
                name="Cypress",
                description="Cypress end-to-end testing",
                weight=0.8,
                required_files=["cypress.json", "cypress.config.js"],
                directory_patterns=["cypress/"]
            )
        }
    
    @staticmethod
    def get_deployment_tool_rules() -> Dict[str, DetectionRule]:
        """Get rules for detecting deployment tools."""
        return {
            "docker": DetectionRule(
                name="Docker",
                description="Docker containerization",
                weight=0.9,
                required_files=["Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"]
            ),
            "kubernetes": DetectionRule(
                name="Kubernetes",
                description="Kubernetes orchestration",
                weight=0.9,
                required_files=["*.yaml", "*.yml"],
                required_patterns=[r"kind:\s*(Deployment|Service|Pod|Ingress)", r"apiVersion:\s*v1"],
                directory_patterns=["k8s/", "kubernetes/"]
            ),
            "terraform": DetectionRule(
                name="Terraform",
                description="Terraform infrastructure as code",
                weight=0.9,
                required_files=["*.tf", "terraform.tfvars"],
                required_patterns=[r"resource\s+\"", r"provider\s+\""]
            ),
            "heroku": DetectionRule(
                name="Heroku",
                description="Heroku deployment platform",
                weight=0.8,
                required_files=["Procfile", "app.json"]
            ),
            "vercel": DetectionRule(
                name="Vercel",
                description="Vercel deployment platform",
                weight=0.8,
                required_files=["vercel.json", ".vercel/"]
            ),
            "netlify": DetectionRule(
                name="Netlify",
                description="Netlify deployment platform",
                weight=0.8,
                required_files=["netlify.toml", "_redirects", ".netlify/"]
            ),
            "github_actions": DetectionRule(
                name="GitHub Actions",
                description="GitHub Actions CI/CD",
                weight=0.8,
                required_files=[".github/workflows/*.yml", ".github/workflows/*.yaml"],
                directory_patterns=[".github/workflows/"]
            ),
            "gitlab_ci": DetectionRule(
                name="GitLab CI",
                description="GitLab CI/CD",
                weight=0.8,
                required_files=[".gitlab-ci.yml"]
            )
        }

# ============================================================================
# Main Service
# ============================================================================

class ProjectTypeDetectionService:
    """
    Autonomous project type detection service using multi-signal analysis.
    
    Provides comprehensive project analysis including:
    - Programming language detection
    - Framework identification  
    - Build tool detection
    - Testing framework identification
    - Deployment tool detection
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the service.
        
        Args:
            llm_provider: Optional LLM provider for advanced analysis
        """
        self.llm_provider = llm_provider
        self.rules = {
            "languages": ProjectTypeDetectionRules.get_language_rules(),
            "frameworks": ProjectTypeDetectionRules.get_framework_rules(),
            "build_tools": ProjectTypeDetectionRules.get_build_tool_rules(),
            "testing_frameworks": ProjectTypeDetectionRules.get_testing_framework_rules(),
            "deployment_tools": ProjectTypeDetectionRules.get_deployment_tool_rules()
        }
        
        logger.info("ProjectTypeDetectionService initialized with multi-signal analysis")
    
    def detect_project_type(
        self,
        project_path: Path,
        max_depth: int = 5,
        enable_content_analysis: bool = True,
        confidence_threshold: float = 0.3
    ) -> ProjectTypeDetectionResult:
        """
        Detect project type and characteristics.
        
        Args:
            project_path: Path to the project root directory
            max_depth: Maximum directory depth to scan
            enable_content_analysis: Whether to analyze file contents
            confidence_threshold: Minimum confidence threshold for characteristics
            
        Returns:
            Complete project type detection result
            
        Raises:
            ProjectTypeDetectionError: If detection fails
        """
        try:
            logger.info(f"Starting project type detection for: {project_path}")
            
            # Gather project information
            project_info = self._gather_project_info(project_path, max_depth)
            
            # Detect characteristics using rules
            all_characteristics = []
            
            for category, rules in self.rules.items():
                characteristics = self._apply_detection_rules(
                    project_info, rules, category, enable_content_analysis
                )
                all_characteristics.extend(characteristics)
            
            # Filter by confidence threshold
            filtered_characteristics = [
                char for char in all_characteristics 
                if char.confidence >= confidence_threshold
            ]
            
            # Determine primary language
            primary_language, language_confidence = self._determine_primary_language(
                filtered_characteristics, project_info
            )
            
            # Categorize characteristics
            frameworks = [c for c in filtered_characteristics if c.category == "frameworks"]
            build_tools = [c for c in filtered_characteristics if c.category == "build_tools"]
            testing_frameworks = [c for c in filtered_characteristics if c.category == "testing_frameworks"]
            deployment_tools = [c for c in filtered_characteristics if c.category == "deployment_tools"]
            
            # Detect project structure type
            structure_type = self._detect_project_structure(project_info, primary_language)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                language_confidence, filtered_characteristics
            )
            
            result = ProjectTypeDetectionResult(
                primary_language=primary_language,
                language_confidence=language_confidence,
                frameworks=frameworks,
                build_tools=build_tools,
                testing_frameworks=testing_frameworks,
                deployment_tools=deployment_tools,
                project_structure_type=structure_type,
                config_files=project_info["config_files"],
                characteristics=filtered_characteristics,
                overall_confidence=overall_confidence,
                detection_metadata={
                    "analyzer_version": "1.0.0",
                    "project_path": str(project_path),
                    "max_depth_scanned": max_depth,
                    "files_analyzed": len(project_info["all_files"]),
                    "content_analysis_enabled": enable_content_analysis
                }
            )
            
            logger.info(
                f"Project type detection completed. Primary language: {primary_language} "
                f"(confidence: {language_confidence:.2f}), {len(filtered_characteristics)} characteristics"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Project type detection failed: {e}")
            raise ProjectTypeDetectionError(f"Detection failed: {e}") from e
    
    def _gather_project_info(
        self, 
        project_path: Path, 
        max_depth: int
    ) -> Dict[str, Any]:
        """Gather information about the project structure and files."""
        info = {
            "all_files": [],
            "file_extensions": {},
            "config_files": [],
            "directories": [],
            "file_contents": {},  # Limited content analysis
            "total_files": 0
        }
        
        try:
            # Collect all files and directories
            for item in project_path.rglob("*"):
                # Skip hidden files and common excluded directories
                if any(part.startswith('.') for part in item.parts):
                    if not any(keep in item.name for keep in [
                        '.github', '.gitignore', '.dockerignore', '.env'
                    ]):
                        continue
                
                if any(skip_dir in item.parts for skip_dir in [
                    'node_modules', '__pycache__', '.git', 'dist', 'build', 
                    '.venv', 'venv', 'env', '.mypy_cache', '.pytest_cache'
                ]):
                    continue
                
                # Check depth limit
                relative_path = item.relative_to(project_path)
                if len(relative_path.parts) > max_depth:
                    continue
                
                if item.is_file():
                    info["all_files"].append(str(item))
                    info["total_files"] += 1
                    
                    # Track file extensions
                    extension = item.suffix.lower()
                    if extension:
                        extension = extension[1:]  # Remove the dot
                        info["file_extensions"][extension] = info["file_extensions"].get(extension, 0) + 1
                    
                    # Identify config files
                    if self._is_config_file(item):
                        info["config_files"].append(str(item))
                
                elif item.is_dir():
                    info["directories"].append(str(item))
            
            logger.debug(f"Gathered info for {info['total_files']} files")
            
        except Exception as e:
            logger.warning(f"Error gathering project info: {e}")
        
        return info
    
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if a file is a configuration file."""
        config_patterns = [
            r".*config\.(js|ts|json|yaml|yml|toml|ini)$",
            r".*\.config\.(js|ts|json|yaml|yml)$",
            r"package\.json$", r"requirements.*\.txt$", r"pyproject\.toml$",
            r"Cargo\.toml$", r"go\.mod$", r"pom\.xml$", r"build\.gradle$",
            r"Dockerfile$", r"docker-compose\.(yml|yaml)$",
            r"tsconfig\.json$", r"jest\.config\.(js|ts)$",
            r"webpack\.config\.(js|ts)$", r"vite\.config\.(js|ts)$",
            r"\.env.*$", r".*\.ini$", r"Makefile$", r"makefile$"
        ]
        
        filename = file_path.name.lower()
        return any(re.match(pattern, filename) for pattern in config_patterns)
    
    def _apply_detection_rules(
        self,
        project_info: Dict[str, Any],
        rules: Dict[str, DetectionRule],
        category: str,
        enable_content_analysis: bool
    ) -> List[ProjectCharacteristic]:
        """Apply detection rules to identify project characteristics."""
        characteristics = []
        
        for rule_name, rule in rules.items():
            confidence = 0.0
            evidence = []
            
            # Check required files
            if rule.required_files:
                file_matches = self._check_file_patterns(
                    project_info["all_files"], rule.required_files
                )
                if file_matches:
                    confidence += 0.4 * rule.weight
                    evidence.extend([f"Found file: {f}" for f in file_matches[:3]])
            
            # Check forbidden files
            if rule.forbidden_files:
                forbidden_matches = self._check_file_patterns(
                    project_info["all_files"], rule.forbidden_files
                )
                if forbidden_matches:
                    confidence -= 0.3 * rule.weight
                    evidence.append(f"Found forbidden file: {forbidden_matches[0]}")
            
            # Check minimum file counts
            if rule.min_file_count:
                for ext, min_count in rule.min_file_count.items():
                    actual_count = project_info["file_extensions"].get(ext, 0)
                    if actual_count >= min_count:
                        confidence += 0.3 * rule.weight
                        evidence.append(f"Found {actual_count} .{ext} files (min: {min_count})")
            
            # Check directory patterns
            if rule.directory_patterns:
                dir_matches = self._check_directory_patterns(
                    project_info["directories"], rule.directory_patterns
                )
                if dir_matches:
                    confidence += 0.2 * rule.weight
                    evidence.extend([f"Found directory: {d}" for d in dir_matches[:2]])
            
            # Check content patterns (if enabled)
            if enable_content_analysis and rule.required_patterns and confidence > 0:
                pattern_matches = self._check_content_patterns(
                    project_info, rule.required_patterns, rule.required_files
                )
                if pattern_matches:
                    confidence += 0.3 * rule.weight
                    evidence.extend([f"Found pattern: {p}" for p in pattern_matches[:2]])
            
            # Normalize confidence
            confidence = min(confidence, 1.0)
            confidence = max(confidence, 0.0)
            
            if confidence > 0:
                characteristic = ProjectCharacteristic(
                    name=rule.name,
                    category=category,
                    confidence=confidence,
                    evidence=evidence,
                    metadata={
                        "rule_name": rule_name,
                        "description": rule.description,
                        "weight": rule.weight
                    }
                )
                characteristics.append(characteristic)
        
        return characteristics
    
    def _check_file_patterns(
        self, 
        file_list: List[str], 
        patterns: List[str]
    ) -> List[str]:
        """Check if files match the given patterns."""
        matches = []
        
        for pattern in patterns:
            # Convert glob pattern to regex
            if '*' in pattern:
                regex_pattern = pattern.replace('*', '.*').replace('?', '.')
                regex_pattern = f".*{regex_pattern}$"
            else:
                regex_pattern = f".*{re.escape(pattern)}$"
            
            for file_path in file_list:
                if re.search(regex_pattern, file_path, re.IGNORECASE):
                    matches.append(file_path)
                    break  # One match per pattern is enough
        
        return matches
    
    def _check_directory_patterns(
        self, 
        dir_list: List[str], 
        patterns: List[str]
    ) -> List[str]:
        """Check if directories match the given patterns."""
        matches = []
        
        for pattern in patterns:
            regex_pattern = pattern.replace('*', '.*').replace('?', '.')
            regex_pattern = f".*{regex_pattern}.*"
            
            for dir_path in dir_list:
                if re.search(regex_pattern, dir_path, re.IGNORECASE):
                    matches.append(dir_path)
                    break
        
        return matches
    
    def _check_content_patterns(
        self,
        project_info: Dict[str, Any],
        patterns: List[str],
        file_patterns: List[str]
    ) -> List[str]:
        """Check if file contents match the given patterns."""
        matches = []
        
        # Find relevant files to analyze
        relevant_files = []
        if file_patterns:
            relevant_files = self._check_file_patterns(
                project_info["all_files"], file_patterns
            )
        else:
            # Analyze some common source files
            relevant_files = [
                f for f in project_info["all_files"][:20]  # Limit to first 20 files
                if any(f.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.rs', '.go'])
            ]
        
        # Check patterns in file contents (limited analysis)
        for file_path in relevant_files[:5]:  # Analyze max 5 files for performance
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(8192)  # Read first 8KB only
                
                for pattern in patterns:
                    if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                        matches.append(pattern)
                        break
                        
            except Exception as e:
                logger.debug(f"Could not analyze content of {file_path}: {e}")
                continue
        
        return matches
    
    def _determine_primary_language(
        self,
        characteristics: List[ProjectCharacteristic],
        project_info: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Determine the primary programming language."""
        language_candidates = [
            c for c in characteristics 
            if c.category == "languages"
        ]
        
        if not language_candidates:
            # Fallback: use file extension analysis
            extensions = project_info["file_extensions"]
            
            # Language mapping by extension
            lang_map = {
                'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                'java': 'java', 'rs': 'rust', 'go': 'go',
                'c': 'c', 'cpp': 'cpp', 'cs': 'csharp',
                'php': 'php', 'rb': 'ruby', 'swift': 'swift'
            }
            
            max_count = 0
            primary_lang = "unknown"
            
            for ext, count in extensions.items():
                if ext in lang_map and count > max_count:
                    max_count = count
                    primary_lang = lang_map[ext]
            
            return primary_lang, 0.5 if primary_lang != "unknown" else 0.1
        
        # Find language with highest confidence
        best_language = max(language_candidates, key=lambda x: x.confidence)
        return best_language.name.lower(), best_language.confidence
    
    def _detect_project_structure(
        self, 
        project_info: Dict[str, Any], 
        primary_language: str
    ) -> Optional[str]:
        """Detect common project structure patterns."""
        all_files = project_info["all_files"]
        
        # Common structure patterns
        if any("src/" in f for f in all_files):
            if any("tests/" in f for f in all_files):
                return "src_tests_structure"
            return "src_structure"
        
        if primary_language == "python":
            if any("setup.py" in f for f in all_files):
                return "python_package"
            if any("__init__.py" in f for f in all_files):
                return "python_module"
        
        if primary_language in ["javascript", "typescript"]:
            if any("pages/" in f for f in all_files):
                return "nextjs_structure"
            if any("components/" in f for f in all_files):
                return "react_structure"
        
        return None
    
    def _calculate_overall_confidence(
        self,
        language_confidence: float,
        characteristics: List[ProjectCharacteristic]
    ) -> float:
        """Calculate overall confidence in the detection."""
        if not characteristics:
            return language_confidence * 0.5
        
        # Weight language confidence heavily
        total_confidence = language_confidence * 0.5
        
        # Add framework/tool confidences
        other_confidences = [c.confidence for c in characteristics if c.category != "languages"]
        if other_confidences:
            avg_other_confidence = sum(other_confidences) / len(other_confidences)
            total_confidence += avg_other_confidence * 0.5
        
        return min(total_confidence, 1.0)

# ============================================================================
# Utility Functions
# ============================================================================

def detect_project_type(
    project_path: Path,
    llm_provider: Optional[LLMProvider] = None,
    max_depth: int = 5
) -> ProjectTypeDetectionResult:
    """
    Convenience function for detecting project type.
    
    Args:
        project_path: Path to the project directory
        llm_provider: Optional LLM provider for advanced analysis
        max_depth: Maximum directory depth to scan
        
    Returns:
        Complete project type detection result
    """
    service = ProjectTypeDetectionService(llm_provider)
    return service.detect_project_type(project_path, max_depth=max_depth)

# Example usage and testing support
if __name__ == "__main__":
    # Test the service
    service = ProjectTypeDetectionService()
    
    # Test with current directory
    result = service.detect_project_type(Path("."))
    print(f"Detection complete: {result.primary_language} project")
    print(f"Frameworks: {[f.name for f in result.frameworks]}")
    print(f"Build tools: {[b.name for b in result.build_tools]}")
    print(f"Overall confidence: {result.overall_confidence:.2f}") 