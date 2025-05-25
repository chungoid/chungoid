# ChromaDB Agent to MCP Tools Migration Plan

## 🎯 **OBJECTIVE**
Replace the obsolete `ProjectChromaManagerAgent_v1` with modern MCP tools for cleaner architecture and better maintainability.

## 🔍 **IMPACT ANALYSIS**

### **Files Requiring Updates:**
- **25+ files** reference `ProjectChromaManagerAgent_v1`
- **8 agent files** directly depend on it for ChromaDB operations
- **Test files** need mock updates
- **CLI commands** need tool integration
- **Orchestrator** needs MCP tool calls instead of agent calls

### **Benefits of Migration:**
1. ✅ **Standardized Interface** - MCP protocol compliance
2. ✅ **Tool Composition** - Can combine with other MCP tools  
3. ✅ **Reduced Complexity** - No agent overhead for data operations
4. ✅ **Better Testing** - Simpler mocking with tools
5. ✅ **Protocol Alignment** - Fits pure protocol architecture

## 🔄 **MIGRATION MAPPING**

| **Agent Method** | **MCP Tool Replacement** | **Notes** |
|------------------|--------------------------|-----------|
| `store_artifact()` | `chromadb_store_document()` | 1:1 replacement |
| `retrieve_artifact()` | `chroma_get_documents()` | 1:1 replacement |
| `get_related_artifacts()` | `chroma_query_documents()` | Query-based approach |
| `initialize_project_collections()` | `chroma_initialize_project_collections()` | 1:1 replacement |
| `log_arca_event()` | `chroma_add_documents()` to ARCA collection | Specialized use case |

## 📋 **MIGRATION STEPS**

### **Phase 1: Core Agent Updates (High Priority)**
1. **Update agent dependencies** - Replace PCMA injection with MCP tool access
2. **Replace storage calls** - Convert `pcma.store_artifact()` to `chromadb_store_document()`
3. **Replace retrieval calls** - Convert `pcma.retrieve_artifact()` to `chroma_get_documents()`
4. **Update initialization** - Use `chroma_initialize_project_collections()`

### **Phase 2: Infrastructure Updates (Medium Priority)**  
1. **Update agent_resolver.py** - Remove PCMA instantiation logic
2. **Update orchestrator.py** - Replace PCMA calls with MCP tool execution
3. **Update CLI commands** - Replace PCMA instantiation with tool calls
4. **Update schemas** - Remove PCMA-specific references

### **Phase 3: Test Updates (Low Priority)**
1. **Mock MCP tools** instead of PCMA agent
2. **Update test fixtures** to use tool mocks
3. **Simplify test setup** without agent instantiation

## 🛠️ **IMPLEMENTATION STRATEGY**

### **Option A: Gradual Migration (RECOMMENDED)**
- Migrate one agent at a time
- Keep compatibility during transition
- Test each migration thoroughly
- Low risk, incremental progress

### **Option B: Big Bang Migration**
- Update all references simultaneously  
- Higher risk but faster completion
- Requires comprehensive testing
- Best for clean slate approach

## 🚧 **CURRENT STATUS**
- ❌ **ProjectChromaManagerAgent_v1 DELETED** (point of no return)
- 🔴 **Build will fail** until references are updated
- 🎯 **Must proceed with migration** to restore functionality

## 📝 **NEXT ACTIONS**
1. Choose migration strategy (recommend Option A)
2. Start with highest priority agent (ArchitectAgent)
3. Update dependencies and test incrementally
4. Document changes for future reference

---
*This migration aligns with our Pure Protocol Architecture goals by eliminating agent overhead for simple data operations.* 