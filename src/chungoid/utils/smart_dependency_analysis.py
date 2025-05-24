"""Smart Dependency Analysis Service

This module provides autonomous dependency analysis capabilities for code projects,
using syntax-based import detection combined with LLM reasoning for intelligent
package mapping and version selection.

Key Features:
- Pure syntax-based import detection for any programming language
- LLM-powered semantic analysis (standard library detection, package mapping)
- Language-agnostic architecture - no hardcoded module knowledge
- Intelligent version constraint recommendation
- Conflict detection and resolution
- Multi-format output generation

Design Principles:
- Autonomous operation with minimal manual configuration
- Separation of syntax parsing from semantic understanding
- Language-agnostic architecture for universal extension
- LLM handles all semantic reasoning about packages and ecosystems
- Zero hardcoded knowledge of specific frameworks or modules
"""

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from pydantic import BaseModel, Field

from .llm_provider import LLMProvider
from .exceptions import ChungoidError
from .project_type_detection import ProjectTypeDetectionService

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================

class DependencyAnalysisError(ChungoidError):
    """Raised when dependency analysis operations fail."""
    pass

class UnsupportedProjectTypeError(DependencyAnalysisError):
    """Raised when attempting to analyze unsupported project types."""
    pass

class PackageMappingError(DependencyAnalysisError):
    """Raised when LLM-based package mapping fails."""
    pass

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ImportInfo:
    """Information about a detected import statement (pure syntax data)."""
    module_name: str
    import_type: str  # 'detected' (all imports are detected, LLM will classify)
    source_file: str
    line_number: int
    is_from_import: bool = False
    imported_names: Optional[List[str]] = None
    
class ProjectType(BaseModel):
    """Detected project type information."""
    language: str = Field(..., description="Primary programming language")
    framework: Optional[str] = Field(None, description="Detected framework (LLM-determined)")
    build_tool: Optional[str] = Field(None, description="Build tool (LLM-determined)")
    config_files: List[str] = Field(default_factory=list, description="Found configuration files")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")

class DependencyInfo(BaseModel):
    """Information about a resolved dependency."""
    package_name: str = Field(..., description="Official package name")
    version_constraint: Optional[str] = Field(None, description="Version constraint (e.g., >=2.0)")
    import_name: str = Field(..., description="Import name used in code")
    description: Optional[str] = Field(None, description="Package description")
    confidence: float = Field(..., description="Mapping confidence 0.0-1.0")
    is_dev_dependency: bool = Field(False, description="Whether this is a development dependency")

class DependencyAnalysisResult(BaseModel):
    """Complete result of dependency analysis."""
    project_type: ProjectType
    dependencies: List[DependencyInfo]
    import_stats: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    generated_files: Dict[str, str]  # filename -> content
    conflicts: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# ============================================================================
# Language Analyzers (Pure Syntax Parsing)
# ============================================================================

class LanguageAnalyzer(ABC):
    """Abstract base class for language-specific syntax analyzers.
    
    These analyzers focus ONLY on syntax parsing - extracting import statements
    from source code. All semantic understanding (standard library detection,
    package mapping) is handled by the LLM.
    """
    
    @abstractmethod
    def extract_imports(self, file_path: Path) -> List[ImportInfo]:
        """Extract import statements from a source file using syntax parsing."""
        pass
    
    @abstractmethod
    def generate_dependency_file(self, dependencies: List[DependencyInfo]) -> Dict[str, str]:
        """Generate dependency files for this language."""
        pass

class PythonAnalyzer(LanguageAnalyzer):
    """Pure syntax analyzer for Python using AST parsing.
    
    Extracts import statements without making any semantic decisions
    about what's standard library vs third-party. The LLM handles all
    semantic classification.
    """
    
    def extract_imports(self, file_path: Path) -> List[ImportInfo]:
        """Extract imports from Python file using AST parsing."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                return imports
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]  # Get top-level module
                        imports.append(ImportInfo(
                            module_name=module_name,
                            import_type='detected',  # LLM will classify
                            source_file=str(file_path),
                            line_number=node.lineno,
                            is_from_import=False,
                            imported_names=[alias.name]
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]  # Get top-level module
                        imported_names = [alias.name for alias in node.names]
                        imports.append(ImportInfo(
                            module_name=module_name,
                            import_type='detected',  # LLM will classify
                            source_file=str(file_path),
                            line_number=node.lineno,
                            is_from_import=True,
                            imported_names=imported_names
                        ))
        
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
        
        return imports
    
    def generate_dependency_file(self, dependencies: List[DependencyInfo]) -> Dict[str, str]:
        """Generate requirements.txt and pyproject.toml content."""
        files = {}
        
        # Generate requirements.txt
        req_lines = []
        dev_req_lines = []
        
        for dep in dependencies:
            if dep.is_dev_dependency:
                line = dep.package_name
                if dep.version_constraint:
                    line += dep.version_constraint
                dev_req_lines.append(line)
            else:
                line = dep.package_name
                if dep.version_constraint:
                    line += dep.version_constraint
                req_lines.append(line)
        
        if req_lines:
            files['requirements.txt'] = '\n'.join(sorted(req_lines)) + '\n'
        
        if dev_req_lines:
            files['requirements-dev.txt'] = '\n'.join(sorted(dev_req_lines)) + '\n'
        
        # Generate basic pyproject.toml
        if req_lines or dev_req_lines:
            toml_content = '[build-system]\n'
            toml_content += 'requires = ["setuptools>=45", "wheel"]\n'
            toml_content += 'build-backend = "setuptools.build_meta"\n\n'
            
            toml_content += '[project]\n'
            toml_content += 'name = "your-project"\n'
            toml_content += 'version = "0.1.0"\n'
            toml_content += 'description = "Auto-generated project configuration"\n'
            
            if req_lines:
                toml_content += 'dependencies = [\n'
                for dep in dependencies:
                    if not dep.is_dev_dependency:
                        line = f'    "{dep.package_name}'
                        if dep.version_constraint:
                            line += dep.version_constraint
                        line += '",'
                        toml_content += line + '\n'
                toml_content += ']\n'
            
            if dev_req_lines:
                toml_content += '\n[project.optional-dependencies]\n'
                toml_content += 'dev = [\n'
                for dep in dependencies:
                    if dep.is_dev_dependency:
                        line = f'    "{dep.package_name}'
                        if dep.version_constraint:
                            line += dep.version_constraint
                        line += '",'
                        toml_content += line + '\n'
                toml_content += ']\n'
            
            files['pyproject.toml'] = toml_content
        
        return files

class NodeJSAnalyzer(LanguageAnalyzer):
    """Pure syntax analyzer for Node.js projects.
    
    Extracts import/require statements without making semantic decisions
    about what's builtin vs third-party. The LLM handles classification.
    """
    
    def extract_imports(self, file_path: Path) -> List[ImportInfo]:
        """Extract imports from JavaScript/TypeScript file using regex parsing."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Patterns for different import styles
            patterns = [
                # require() statements
                r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
                # ES6 import statements
                r"import\s+(?:(?:\w+|\{[^}]+\}|\*\s+as\s+\w+)(?:\s*,\s*(?:\w+|\{[^}]+\}|\*\s+as\s+\w+))*\s+from\s+)?['\"]([^'\"]+)['\"]",
                # dynamic imports
                r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
            ]
            
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        module_name = match.group(1)
                        # Extract package name (before first '/')
                        package_name = module_name.split('/')[0]
                        if package_name.startswith('@'):
                            # Scoped package
                            parts = module_name.split('/')
                            if len(parts) > 1:
                                package_name = f"{parts[0]}/{parts[1]}"
                        
                        imports.append(ImportInfo(
                            module_name=package_name,
                            import_type='detected',  # LLM will classify
                            source_file=str(file_path),
                            line_number=i,
                            is_from_import=False,
                            imported_names=[module_name]
                        ))
        
        except Exception as e:
            logger.error(f"Error analyzing Node.js file {file_path}: {e}")
        
        return imports
    
    def generate_dependency_file(self, dependencies: List[DependencyInfo]) -> Dict[str, str]:
        """Generate package.json content."""
        package_json = {
            "name": "your-project",
            "version": "1.0.0",
            "description": "Auto-generated project",
            "main": "index.js",
            "dependencies": {},
            "devDependencies": {}
        }
        
        for dep in dependencies:
            version = dep.version_constraint or "^1.0.0"
            if dep.is_dev_dependency:
                package_json["devDependencies"][dep.package_name] = version
            else:
                package_json["dependencies"][dep.package_name] = version
        
        return {"package.json": json.dumps(package_json, indent=2)}

# ============================================================================
# Main Service
# ============================================================================

class SmartDependencyAnalysisService:
    """
    Language-agnostic dependency analysis service combining syntax parsing with LLM intelligence.
    
    Architecture:
    1. Language analyzers perform pure syntax parsing (extract imports)
    2. LLM handles all semantic understanding (standard library detection, package mapping)
    3. No hardcoded knowledge of specific modules or frameworks
    4. Extensible to any language by adding syntax analyzer
    
    Provides comprehensive dependency analysis for software projects including:
    - Multi-language import detection 
    - LLM-powered package name mapping and classification
    - Version constraint recommendation
    - Conflict detection and resolution
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize the service with LLM provider for intelligent reasoning.
        
        Args:
            llm_provider: LLM provider for package mapping and semantic analysis
        """
        self.llm_provider = llm_provider
        self.project_type_detector = ProjectTypeDetectionService(llm_provider)
        self.analyzers = {
            'python': PythonAnalyzer(),
            'javascript': NodeJSAnalyzer(),
            'typescript': NodeJSAnalyzer(),  # TypeScript uses same syntax analyzer as JS
            # Adding new languages is now trivial - just implement syntax parsing:
            # 'rust': RustAnalyzer(),      # Parse 'use' statements
            # 'go': GoAnalyzer(),          # Parse 'import' statements  
            # 'java': JavaAnalyzer(),      # Parse 'import' statements
            # 'csharp': CSharpAnalyzer(),  # Parse 'using' statements
        }
        
        logger.info("SmartDependencyAnalysisService initialized with language-agnostic LLM-powered analysis")
    
    async def analyze_project(
        self,
        project_path: Path,
        project_type: Optional[str] = None,
        include_dev_dependencies: bool = True,
        max_depth: int = 10
    ) -> DependencyAnalysisResult:
        """
        Perform comprehensive dependency analysis on a project.
        
        Args:
            project_path: Path to the project root directory
            project_type: Optional explicit project type (python, javascript, typescript)
            include_dev_dependencies: Whether to include development dependencies
            max_depth: Maximum directory depth to scan
            
        Returns:
            Complete dependency analysis result
            
        Raises:
            DependencyAnalysisError: If analysis fails
            UnsupportedProjectTypeError: If project type cannot be analyzed
        """
        try:
            logger.info(f"Starting dependency analysis for project: {project_path}")
            
            # Detect project type if not specified
            if not project_type:
                detection_result = self.project_type_detector.detect_project_type(project_path)
                project_type = detection_result.primary_language
                detected_type = ProjectType(
                    language=detection_result.primary_language,
                    framework=detection_result.frameworks[0].name if detection_result.frameworks else None,
                    build_tool=detection_result.build_tools[0].name if detection_result.build_tools else None,
                    config_files=detection_result.config_files,
                    confidence=detection_result.language_confidence
                )
            else:
                detected_type = ProjectType(
                    language=project_type,
                    confidence=1.0
                )
            
            if project_type not in self.analyzers:
                raise UnsupportedProjectTypeError(
                    f"Unsupported project type: {project_type}. "
                    f"Supported types: {list(self.analyzers.keys())}"
                )
            
            analyzer = self.analyzers[project_type]
            
            # Extract all imports from source files
            all_imports = self._extract_all_imports(
                project_path, analyzer, max_depth
            )
            
            # Get unique module names (LLM will filter standard library)
            unique_modules = list(set(imp.module_name for imp in all_imports))
            
            logger.info(f"Found {len(unique_modules)} unique imported modules (LLM will filter standard library)")
            
            # Use LLM to map import names to package names
            dependencies = await self._resolve_dependencies_with_llm(
                unique_modules, project_type, include_dev_dependencies
            )
            
            # Generate dependency files
            generated_files = analyzer.generate_dependency_file(dependencies)
            
            # Calculate statistics
            import_stats = self._calculate_import_stats(all_imports)
            
            # Detect potential conflicts
            conflicts = self._detect_conflicts(dependencies)
            
            result = DependencyAnalysisResult(
                project_type=detected_type,
                dependencies=dependencies,
                import_stats=import_stats,
                analysis_metadata={
                    "analyzer_version": "1.0.0",
                    "project_path": str(project_path),
                    "analysis_timestamp": "auto-generated",
                    "total_files_analyzed": import_stats.get("files_analyzed", 0),
                    "llm_mappings_used": len(dependencies)
                },
                generated_files=generated_files,
                conflicts=conflicts,
                warnings=[]
            )
            
            logger.info(f"Dependency analysis completed successfully. Found {len(dependencies)} dependencies")
            return result
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            raise DependencyAnalysisError(f"Analysis failed: {e}") from e
    

    
    def _extract_all_imports(
        self, 
        project_path: Path, 
        analyzer: LanguageAnalyzer, 
        max_depth: int
    ) -> List[ImportInfo]:
        """Extract imports from all relevant source files in the project."""
        all_imports = []
        files_analyzed = 0
        
        # File extensions by language
        extensions = {
            'python': ['*.py'],
            'javascript': ['*.js', '*.jsx'],
            'typescript': ['*.ts', '*.tsx', '*.js', '*.jsx']
        }
        
        # Determine which extensions to look for
        if isinstance(analyzer, PythonAnalyzer):
            file_patterns = extensions['python']
        elif isinstance(analyzer, NodeJSAnalyzer):
            file_patterns = extensions['javascript'] + extensions['typescript']
        else:
            file_patterns = ['*']
        
        # Find and analyze source files
        for pattern in file_patterns:
            for file_path in project_path.rglob(pattern):
                # Skip certain directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(skip_dir in file_path.parts for skip_dir in [
                    'node_modules', '__pycache__', '.git', 'dist', 'build'
                ]):
                    continue
                
                # Check depth limit
                relative_path = file_path.relative_to(project_path)
                if len(relative_path.parts) > max_depth:
                    continue
                
                try:
                    imports = analyzer.extract_imports(file_path)
                    all_imports.extend(imports)
                    files_analyzed += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
        
        logger.info(f"Analyzed {files_analyzed} source files, found {len(all_imports)} imports")
        return all_imports
    
    async def _resolve_dependencies_with_llm(
        self,
        module_names: List[str],
        project_type: str,
        include_dev_dependencies: bool
    ) -> List[DependencyInfo]:
        """Use LLM to perform semantic analysis of imports - classify and map to packages."""
        if not module_names:
            return []
        
        # Prepare LLM prompt for semantic analysis
        system_prompt = f"""You are an expert software developer with deep knowledge of {project_type} ecosystems and dependency management.

Your task is to analyze import/require statements and:
1. Filter out standard library modules (built-in to the language/runtime)
2. For remaining third-party imports, map them to official package names
3. Recommend appropriate version constraints
4. Classify as runtime vs development dependencies

Return ONLY third-party packages that need to be installed via package managers.
Exclude all standard library modules, built-ins, and local project imports.

Return your response as valid JSON:
{{
  "mappings": [
    {{
      "import_name": "imported_name",
      "package_name": "official_package_name",
      "version_constraint": "version_constraint_or_null",
      "is_dev_dependency": boolean,
      "confidence": 0.0_to_1.0,
      "description": "brief_description"
    }}
  ]
}}

Be conservative with version constraints and accurate with package names.
Set confidence < 0.5 for uncertain mappings."""
        
        user_prompt = f"""Analyze these {project_type} import names:

{', '.join(module_names)}

Instructions:
- Language: {project_type}
- Include dev dependencies: {include_dev_dependencies}
- Filter out standard library modules
- Return only third-party packages that need installation
- Use official package manager names for your target language

Return the analysis in the specified JSON format."""
        
        try:
            logger.info(f"Requesting LLM mapping for {len(module_names)} modules")
            
            response = await self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for consistent results
                response_format={"type": "json_object"}  # Request JSON response
            )
            
            # Parse LLM response
            mapping_data = json.loads(response)
            dependencies = []
            
            for mapping in mapping_data.get("mappings", []):
                dep = DependencyInfo(
                    package_name=mapping["package_name"],
                    version_constraint=mapping.get("version_constraint"),
                    import_name=mapping["import_name"],
                    description=mapping.get("description"),
                    confidence=mapping.get("confidence", 0.5),
                    is_dev_dependency=mapping.get("is_dev_dependency", False)
                )
                dependencies.append(dep)
            
            logger.info(f"Successfully mapped {len(dependencies)} dependencies using LLM")
            return dependencies
            
        except Exception as e:
            logger.error(f"LLM dependency mapping failed: {e}")
            
            # Fallback: create basic dependencies without LLM
            logger.info("Using fallback dependency mapping")
            dependencies = []
            for module_name in module_names:
                dep = DependencyInfo(
                    package_name=module_name,  # Assume package name = import name
                    version_constraint=None,   # No version constraint
                    import_name=module_name,
                    description="Auto-detected dependency",
                    confidence=0.3,  # Low confidence for fallback
                    is_dev_dependency=False
                )
                dependencies.append(dep)
            
            return dependencies
    
    def _calculate_import_stats(self, imports: List[ImportInfo]) -> Dict[str, Any]:
        """Calculate statistics about the imports found."""
        stats = {
            "total_imports": len(imports),
            "files_analyzed": len(set(imp.source_file for imp in imports)),
            "import_types": {}
        }
        
        # Count by import type
        for import_type in ['standard', 'local', 'third_party']:
            count = sum(1 for imp in imports if imp.import_type == import_type)
            stats["import_types"][import_type] = count
        
        # Count unique modules
        unique_modules = set(imp.module_name for imp in imports)
        stats["unique_modules"] = len(unique_modules)
        
        return stats
    
    def _detect_conflicts(self, dependencies: List[DependencyInfo]) -> List[str]:
        """Detect potential dependency conflicts."""
        conflicts = []
        
        # Check for duplicate package names with different constraints
        package_constraints = {}
        for dep in dependencies:
            if dep.package_name in package_constraints:
                existing_constraint = package_constraints[dep.package_name]
                if existing_constraint != dep.version_constraint:
                    conflicts.append(
                        f"Conflicting version constraints for {dep.package_name}: "
                        f"{existing_constraint} vs {dep.version_constraint}"
                    )
            else:
                package_constraints[dep.package_name] = dep.version_constraint
        
        return conflicts

# ============================================================================
# Utility Functions
# ============================================================================

async def analyze_project_dependencies(
    project_path: Union[str, Path],
    llm_provider: LLMProvider,
    project_type: Optional[str] = None
) -> DependencyAnalysisResult:
    """
    Convenience function for analyzing project dependencies.
    
    Args:
        project_path: Path to the project directory
        llm_provider: LLM provider for intelligent analysis
        project_type: Optional explicit project type
        
    Returns:
        Complete dependency analysis result
    """
    service = SmartDependencyAnalysisService(llm_provider)
    return await service.analyze_project(
        Path(project_path), 
        project_type=project_type
    )

# Example usage and testing support
if __name__ == "__main__":
    # This would be used for testing the service
    import asyncio
    from .llm_provider import MockLLMProvider
    
    async def test_analysis():
        # Mock LLM provider for testing
        mock_llm = MockLLMProvider({
            "Map these python import names": json.dumps({
                "mappings": [
                    {
                        "import_name": "requests", 
                        "package_name": "requests", 
                        "version_constraint": ">=2.25.0",
                        "is_dev_dependency": False,
                        "confidence": 0.95,
                        "description": "HTTP library"
                    }
                ]
            })
        })
        
        service = SmartDependencyAnalysisService(mock_llm)
        
        # Test with current directory
        result = await service.analyze_project(Path("."))
        print(f"Analysis complete: {len(result.dependencies)} dependencies found")
        
    # Run test if executed directly
    asyncio.run(test_analysis()) 