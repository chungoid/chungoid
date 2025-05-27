# Documentation Maintenance Guide

*Best practices for keeping Chungoid documentation current, accurate, and useful.*

## Overview

This guide outlines the documentation maintenance strategy for Chungoid, implementing best practices from leading software organizations to keep documentation synchronized and valuable.

### Documentation Philosophy

Following recommendations from [JetBrains](https://blog.jetbrains.com/writerside/2022/01/the-holy-grail-of-always-up-to-date-documentation/) and [Archbee](https://www.archbee.com/blog/keeping-product-documentation-up-to-date/), our approach focuses on:

- **Action-based writing**: Focus on user scenarios and problem-solving rather than descriptive details
- **Automated synchronization**: Keep documentation close to code with automated updates
- **Living documentation**: Regular updates integrated into development workflow
- **User-centric organization**: Structure for different audiences (developers, users, integrators)

## Documentation Architecture

### Source Structure (`dev/docs/`)
The authoritative documentation source lives in `dev/docs/` alongside the meta-layer development environment:

```
dev/docs/
├── architecture/           # System architecture and design decisions
├── system_design/         # Detailed component design documents  
├── guides/               # User and developer guides
├── reference/            # API and schema reference
├── learnings/            # Development insights and lessons learned
└── *.md                  # Core overview documents
```

### Target Structure (`chungoid-core/docs/`)
User-facing documentation synchronized to `chungoid-core/docs/`:

```
chungoid-core/docs/
├── architecture/         # Adapted architecture docs for users
├── design_documents/     # Agent and component designs
├── guides/              # User guides adapted from source
├── reference/           # Reference documentation
├── sync_report.md       # Latest synchronization report
└── *.sync.json         # Synchronization metadata
```

## Automated Synchronization

### Sync Script Usage

The `scripts/sync_documentation.py` script automatically synchronizes documentation:

```bash
# Dry run to see what would be updated
python scripts/sync_documentation.py --dry-run

# Synchronize all updated files
python scripts/sync_documentation.py

# Force update all files regardless of timestamps
python scripts/sync_documentation.py --force
```

### Sync Operations

The script performs three types of operations:

1. **Copy**: Direct copy of files that don't need adaptation
2. **Adapt**: Transform content for different audiences (e.g., user guides vs. internal docs)
3. **Replace**: Complete replacement with migration notes

### Sync Metadata

Each synchronized file includes:
- Source file reference
- Synchronization timestamp
- Transform type applied
- Sync metadata in adjacent `.sync.json` files

## Maintenance Workflow

### Regular Updates

**Weekly Sync** (Recommended):
```bash
# Check what needs updating
python scripts/sync_documentation.py --dry-run

# Apply updates
python scripts/sync_documentation.py

# Review sync report
cat chungoid-core/docs/sync_report.md
```

### Development Integration

**After Major Changes**:
1. Update source documentation in `dev/docs/`
2. Run synchronization script
3. Review adapted content for accuracy
4. Commit both source and synchronized documentation

**Continuous Integration**:
Consider adding sync validation to CI/CD:
```bash
# Check if sync is needed
python scripts/sync_documentation.py --dry-run --check-only
```

## Content Guidelines

### Action-Based Writing

Focus on what users can **do** rather than descriptive details:

❌ **Descriptive**: "The `UnifiedOrchestrator` class has a run() method that takes an ExecutionPlan parameter"

✅ **Action-Based**: "To execute a workflow, pass your ExecutionPlan to `orchestrator.run()`"

### Audience Adaptation

Content automatically adapts for different audiences:

- **User Guides**: Practical usage, simplified technical details
- **Reference Docs**: Complete technical details for lookup
- **Architecture Docs**: Design decisions and system interactions

### Migration Notes

When replacing outdated content, include migration context:

```markdown
> **Migration Note**: This document has been updated to reflect the current 
> system architecture. Previous concepts have evolved into [new approach].
```

## Quality Assurance

### Validation Checklist

Before publishing updates:

- [ ] Source documentation accurately reflects current system
- [ ] Adapted content maintains technical accuracy
- [ ] User guides focus on practical scenarios
- [ ] Reference documentation is complete
- [ ] Migration notes explain changes clearly

### Feedback Integration

**User Feedback**:
- Monitor documentation usage analytics
- Collect feedback on clarity and completeness
- Update source documentation based on common questions

**Developer Feedback**:
- Review sync results for technical accuracy
- Validate that adaptations preserve important details
- Ensure new features are properly documented

## Troubleshooting

### Common Issues

**Sync Conflicts**:
```bash
# Force update if timestamps are problematic
python scripts/sync_documentation.py --force
```

**Missing Source Files**:
Check that source documentation exists in `dev/docs/` and update sync mappings if structure changes.

**Content Adaptation Issues**:
Review adaptation logic in `DocumentationSyncer._adapt_for_*()` methods and adjust for specific content types.

### Recovery

**Restore from Backup**:
The sync script creates `.backup` files when replacing content:
```bash
# Restore original file
cp chungoid-core/docs/filename.md.backup chungoid-core/docs/filename.md
```

**Manual Override**:
Temporarily modify sync mappings to skip problematic files or use different adaptation strategies.

## Best Practices

### Keep Documentation Close to Code

✅ **Do**:
- Update documentation when changing related code
- Include documentation updates in feature branches
- Reference code examples from actual repository files

❌ **Don't**:
- Let documentation lag behind code changes
- Hardcode examples that can become outdated
- Write documentation in isolation from development

### Focus on User Value

✅ **Do**:
- Solve user problems with step-by-step guidance
- Provide working examples and common use cases
- Explain the "why" behind architectural decisions

❌ **Don't**:
- Document every implementation detail
- Write purely descriptive content without actionable guidance
- Assume users understand internal system complexity

### Maintain Consistency

✅ **Do**:
- Use consistent terminology across all documentation
- Follow established style and formatting guidelines
- Maintain standard structure across similar document types

❌ **Don't**:
- Use different terms for the same concepts
- Mix detailed technical specs with high-level overviews
- Create inconsistent navigation or organization

## Future Enhancements

### Planned Improvements

1. **CI/CD Integration**: Automatic sync validation in pull requests
2. **Content Analytics**: Track documentation usage and effectiveness
3. **Interactive Examples**: Integrate runnable code examples
4. **Version Alignment**: Ensure documentation versions match code releases

### Monitoring

- **Sync Health**: Regular validation that synchronization is working correctly
- **Content Freshness**: Alerts when source documentation hasn't been updated recently
- **User Experience**: Feedback collection and analysis for continuous improvement

---

*For questions about documentation maintenance, refer to the sync script source code or create an issue for process improvements.* 