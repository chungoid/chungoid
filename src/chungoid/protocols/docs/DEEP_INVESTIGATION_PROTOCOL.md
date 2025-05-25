# Deep Investigation Protocol

## Document Status
- **Created**: 2025-01-24
- **Updated**: 2025-01-24
- **Purpose**: Ensure maximum thoroughness in problem analysis before solution implementation
- **Scope**: All complex technical issues requiring multi-step solutions
- **Proven Success**: SystemRequirementsGatheringAgent_v1 investigation (2025-01-24)

---

## Core Principle

**"Understand First, Solve Second"**

> *The cost of fixing the wrong problem is always higher than the time spent understanding the right problem.*

**Proven Evidence**: Our investigation revealed agent schema was correct, preventing wrong fix that would have wasted hours and created more problems.

---

## INVESTIGATION PHASES & GATES

### Phase 1: Surface-Level Investigation
**Goal**: Document all visible symptoms and establish reproduction
**Time Box**: 30-60 minutes for most issues

#### Required Outputs:
- [ ] **Complete Error Documentation** - All error messages, stack traces, log entries
- [ ] **Reproduction Steps** - Reliable way to trigger the issue
- [ ] **Component Inventory** - List of all involved files/modules
- [ ] **Initial Timeline** - When issue was first noticed, recent changes
- [ ] **Initial Hypothesis** - First guess at root cause (to test later)

#### Chungoid-Specific Investigation Tools:
- [ ] **`codebase_search`** - Semantic search for relevant code patterns
- [ ] **`grep_search`** - Exact text/regex searches for error messages
- [ ] **`file_search`** - Find related files by name patterns
- [ ] **Terminal reproduction** - Use `run_terminal_cmd` to reproduce issue

#### Gate Criteria:
- Can reproduce issue on demand
- All visible symptoms documented
- Component scope identified
- Initial hypothesis documented (even if wrong)

**❌ COMMON MISTAKE**: Jumping to solutions based on error messages alone
**✅ SUCCESS PATTERN**: Document everything, assume nothing

---

### Phase 2: Deep Root Cause Analysis
**Goal**: Understand WHY the problem exists, not just WHAT is failing
**Time Box**: 1-3 hours depending on complexity

#### Investigation Methods:

##### A. Code Archaeological Dig
- **Read complete source** - Every line of involved components using `read_file`
- **Understand original intent** - Why was it written this way?
- **Document evolution** - Check git history for changes
- **Identify debt** - Workarounds, TODOs, quick fixes

**Proven Technique**: Read entire agent implementation line-by-line revealed correct schema

##### B. Schema Forensics (Python/Pydantic Specific)
- **Runtime data capture** - Log actual data structures at boundaries
- **Pydantic model analysis** - Compare expected vs actual field names/types
- **Type system verification** - Check type hints match runtime behavior
- **Import chain analysis** - Verify all imports resolve correctly

**Success Example**: Found `SystemRequirementsGatheringOutput` schema matched success criteria exactly

##### C. Flow State Analysis
- **Step-through debugging** - Actual execution path
- **State capture** - Variables at each decision point  
- **Context flow tracing** - How data moves between components
- **Integration handoffs** - Data transformation points

**New Focus Area**: Context management and data flow between orchestration layers

##### D. Dependency Web Mapping
- **Direct dependencies** - What this code calls
- **Inverse dependencies** - What calls this code
- **Implicit contracts** - Undocumented assumptions
- **Failure propagation** - How errors cascade

#### Investigation Session Structure:
1. **Session Planning** (10 min) - Define specific investigation goals
2. **Deep Dive** (60-120 min) - Systematic investigation using tools above
3. **Findings Synthesis** (15 min) - Document discoveries and pivot if needed
4. **Next Session Planning** (5 min) - Define next investigation targets

#### Required Outputs:
- [ ] **Complete mental model** - Can explain to someone else
- [ ] **Historical context** - Why current patterns exist
- [ ] **Failure analysis** - Root cause, not symptoms
- [ ] **Alternative research** - How others solve similar problems
- [ ] **Investigation pivot** - Update direction based on findings

#### Gate Criteria:
- Can explain problem without looking at notes
- Understand original design intent
- Know why current approach fails OR discovered initial hypothesis was wrong
- Have researched alternative approaches

**❌ COMMON MISTAKE**: Assuming current implementation is "correct"
**✅ SUCCESS PATTERN**: Question all assumptions, especially initial ones

---

### Phase 3: Comprehensive Understanding
**Goal**: Map full problem space and solution constraints
**Time Box**: 45-90 minutes

#### Investigation Areas:

##### A. Edge Case Analysis
- **Boundary conditions** - What happens at limits?
- **Error scenarios** - All ways this can fail
- **Load conditions** - Performance under stress
- **Data variations** - Different input types/sizes

##### B. Impact Assessment
- **Downstream effects** - What else might break?
- **Performance implications** - Speed, memory, resources
- **Security considerations** - Attack vectors, data exposure
- **Compatibility impact** - Version dependencies, breaking changes

##### C. Constraint Identification
- **Technical constraints** - Language, framework, library limitations
- **Business constraints** - Timeline, resources, scope
- **Legacy constraints** - Existing integrations, data formats
- **Operational constraints** - Deployment, monitoring, support

#### Investigation Templates by Issue Type:

##### Schema/Data Flow Issues:
- [ ] Map complete data transformation pipeline
- [ ] Identify all validation points
- [ ] Check serialization/deserialization boundaries
- [ ] Verify context management patterns

##### Orchestration Issues:
- [ ] Trace agent lifecycle from instantiation to output
- [ ] Map context population and retrieval mechanisms
- [ ] Check success criteria evaluation logic
- [ ] Verify error handling pathways

##### Integration Issues:
- [ ] Check all component interfaces
- [ ] Verify dependency injection patterns
- [ ] Test component isolation
- [ ] Map configuration flow

#### Required Outputs:
- [ ] **Complete problem specification** - All scenarios documented
- [ ] **Constraint catalog** - All limitations identified
- [ ] **Impact map** - Downstream effects documented
- [ ] **Success criteria** - Clear, testable definitions

#### Gate Criteria:
- All edge cases identified
- Full constraint catalog complete
- Success criteria defined and measurable

**❌ COMMON MISTAKE**: Focusing only on "happy path" scenarios
**✅ SUCCESS PATTERN**: Consider all failure modes and edge cases

---

### Phase 4: Solution Design Readiness
**Goal**: Ensure multiple solution paths with trade-off analysis
**Time Box**: 60-90 minutes

#### Solution Design Requirements:

##### A. Multiple Approaches
- **Conservative approach** - Minimal change, lowest risk
- **Optimal approach** - Best long-term solution
- **Pragmatic approach** - Balance of risk/benefit/timeline

##### B. Trade-off Analysis
- **Implementation complexity** - Development effort required
- **Risk assessment** - What could go wrong?
- **Performance impact** - Speed, memory, scalability effects
- **Maintenance burden** - Long-term support costs

##### C. Implementation Planning
- **Step-by-step approach** - Incremental implementation
- **Testing strategy** - How to validate each step
- **Rollback plan** - How to undo if things go wrong
- **Monitoring plan** - How to detect problems in production

#### Solution Design Templates:

##### For Schema/Data Issues:
- **Conservative**: Fix field mapping only
- **Optimal**: Redesign data flow for robustness
- **Pragmatic**: Fix immediate issue + add validation

##### For Orchestration Issues:
- **Conservative**: Patch specific failure point
- **Optimal**: Redesign context management system
- **Pragmatic**: Fix context flow + add debugging

#### Required Outputs:
- [ ] **Multiple solution designs** - At least 2-3 viable approaches
- [ ] **Detailed trade-off analysis** - Pros/cons of each approach
- [ ] **Implementation roadmap** - Step-by-step plan
- [ ] **Risk mitigation plan** - Rollback and monitoring strategy

#### Gate Criteria:
- Multiple solutions with trade-offs documented
- Implementation plan with clear steps
- Risk mitigation strategy defined

**❌ COMMON MISTAKE**: Implementing first solution that comes to mind
**✅ SUCCESS PATTERN**: Design multiple approaches, choose best fit

---

## PROCESS ENFORCEMENT

### Mandatory Checkpoints

1. **Phase Gate Reviews** - Must complete all requirements before advancing
2. **Peer Validation** - Second person reviews analysis before solutions
3. **Documentation Requirements** - All findings documented before proceeding
4. **Solution Approval** - Multiple stakeholders approve approach before implementation

### Enhanced Warning Signs of Insufficient Investigation

- **"Quick fix" mentality** - Jumping to code changes immediately
- **Single solution focus** - Only considering one approach
- **Symptom treatment** - Fixing errors without understanding root cause
- **Missing edge cases** - Only testing "happy path"
- **No rollback plan** - Can't undo changes if needed
- **Assumption persistence** - Sticking with initial hypothesis despite contrary evidence
- **Tool avoidance** - Not using available investigation tools
- **Documentation gaps** - Not capturing investigation process

### Escalation Triggers

**Stop and conduct deeper investigation if:**
- Fix attempt fails or creates new problems
- Multiple components involved across module boundaries
- Issue affects critical path or user-facing functionality
- Problem has unclear scope or expanding symptoms
- Previous "fixes" for similar issues have failed
- Initial hypothesis proven wrong during investigation
- Investigation reveals systemic issues

---

## ENHANCED DOCUMENTATION STANDARDS

### Analysis Documentation Requirements

#### For Each Investigation Phase:
- **Checklist completion** - All items marked as done with timestamps
- **Evidence collection** - Screenshots, logs, data samples, code snippets
- **Source references** - File paths, line numbers, commit hashes
- **Tool usage log** - Which investigation tools were used and what they revealed
- **Hypothesis evolution** - How understanding changed during investigation

#### For Solutions:
- **Design rationale** - Why this approach was chosen
- **Alternative consideration** - What other options were evaluated with specific reasons for rejection
- **Implementation notes** - Key decisions and trade-offs
- **Test validation** - How solution was verified
- **Rollback documentation** - Exact steps to undo changes

### Investigation Templates

#### Schema Investigation Template:
```markdown
## Schema Investigation: [Component Name]

### Expected Schema:
```python
[Pydantic model or type definition]
```

### Actual Runtime Data:
```python
[Logged actual data structure]
```

### Field Mapping Analysis:
- Field X: Expected [type], Found [type]
- Missing fields: [list]
- Extra fields: [list]

### Validation Points:
[Where/how schema validation occurs]
```

#### Context Flow Investigation Template:
```markdown
## Context Flow: [Process Name]

### Data Entry Points:
1. [Where data enters system]
2. [Transformation points]

### Storage Mechanisms:
- context.outputs.[stage].[field]
- [Other storage patterns]

### Retrieval Mechanisms:
- Success criteria: [expression]
- Resolution result: [actual value]

### Gap Analysis:
[Where data is lost/transformed incorrectly]
```

---

## ENHANCED TOOLS & TECHNIQUES

### Chungoid-Specific Investigation Tools
- **`codebase_search`** - Semantic search for patterns and concepts
- **`grep_search`** - Exact regex searches for error patterns
- **`file_search`** - Fuzzy filename matching
- **`read_file`** - Systematic code reading with line ranges
- **`run_terminal_cmd`** - Reproduce issues and test fixes

### Investigation Tool Combinations:
1. **Error Analysis**: `grep_search` for error messages → `codebase_search` for related code → `read_file` for implementation
2. **Schema Analysis**: `codebase_search` for Pydantic models → `read_file` for definitions → `grep_search` for usage patterns
3. **Flow Analysis**: `codebase_search` for orchestration → `read_file` for implementation → terminal testing

### Documentation Tools
- **Structured templates** - Use investigation templates above
- **Evidence collection** - Screenshots, logs, code citations
- **Decision tracking** - Why choices were made at each step
- **Timeline documentation** - When each investigation occurred

---

## QUALITY ASSURANCE

### Enhanced Self-Assessment Questions

Before proceeding to solution implementation, ask:

1. **Can I explain this problem to someone else without notes?**
2. **Do I understand why the current approach fails?**
3. **Have I considered at least 2 different solution approaches?**
4. **Do I know what could go wrong with my proposed solution?**
5. **Can I roll back my changes if the solution fails?**
6. **Have I documented my analysis for future reference?**
7. **Did I question my initial assumptions?** ⭐ NEW
8. **Did I use the appropriate investigation tools?** ⭐ NEW
9. **Would this solution address the root cause or just symptoms?** ⭐ NEW

### Enhanced Peer Review Checklist

When reviewing someone else's analysis:

- [ ] Is the root cause clearly identified?
- [ ] Were initial assumptions tested and validated/refuted?
- [ ] Are all edge cases considered?
- [ ] Are multiple solution approaches documented?
- [ ] Is the implementation plan realistic?
- [ ] Is there a clear rollback strategy?
- [ ] Is the analysis well-documented for future reference?
- [ ] Were appropriate investigation tools used?
- [ ] Does the solution address root cause vs symptoms?

---

## SUCCESS METRICS

### Investigation Quality Indicators
- **Root cause accuracy** - Solutions address actual underlying issues
- **Solution durability** - Fixes don't create new problems
- **Knowledge transfer** - Others can understand and maintain solutions
- **Pattern recognition** - Similar problems avoided in future
- **Assumption validation** - Initial hypotheses tested and corrected

### Process Effectiveness Indicators
- **Reduced rework** - Fewer failed fix attempts
- **Faster resolution** - Time from problem to working solution
- **Better solutions** - More robust, maintainable fixes
- **Team learning** - Improved problem-solving capabilities
- **Investigation pivot success** - Ability to change direction when initial assumptions prove wrong

---

## INTEGRATION WITH 3-STEP PROCESS

This protocol integrates with our existing development process:

1. **Analysis Phase** (`dev/planning/analysis/`) - Follow this investigation protocol completely
2. **Planning Phase** (`dev/planning/blueprints/` & `roadmaps/`) - Only after all 4 phases complete
3. **Archive Phase** (`dev/planning/archive/`) - Include investigation artifacts and lessons learned

**Key Integration Points:**
- Analysis files must pass all phase gates before blueprints created
- Blueprint/roadmap creation requires peer review of analysis
- Implementation only begins after solution design approval
- Archive includes both solution and investigation documentation
- Investigation pivot discoveries update analysis files in real-time

### Analysis File Integration:
- Use investigation templates in analysis files
- Update analysis continuously during investigation
- Mark phase completions with checkboxes and timestamps
- Document hypothesis evolution and pivot points

---

## PROVEN SUCCESS PATTERN

Based on our SystemRequirementsGatheringAgent_v1 investigation:

### What Worked:
1. **Systematic code reading** revealed agent implementation was correct
2. **Schema forensics** showed field names matched expectations  
3. **Phase gates** prevented wrong fix (agent schema modification)
4. **Documentation** captured pivot from agent issues to orchestration issues
5. **Tool usage** - `codebase_search` + `read_file` combination was highly effective

### Lessons Learned:
1. **Question initial assumptions immediately**
2. **Surface-level symptoms often mislead**
3. **Complete code reading beats assumptions**
4. **Schema analysis must compare expected vs actual at runtime**
5. **Investigation pivot is a feature, not a failure**

---

*This protocol ensures maximum understanding before solution implementation, reducing rework and improving solution quality.*

**Remember: The goal is not to delay solutions, but to ensure they solve the right problems correctly.**

**Proven Success**: Prevented wrong schema fix, identified real orchestration issue, saved hours of misdirected effort. 