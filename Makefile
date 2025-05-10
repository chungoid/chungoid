.PHONY: snapshot-core

# Generate a dry-run core snapshot (semantic metadata tarball)
snapshot-core:
	python dev/scripts/embed_core_snapshot.py run --dry-run 