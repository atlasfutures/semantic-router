package config

// Fork seam: rayline_arc joins upstream's typed algorithm registry from this
// file so the registry literal in routing_surface_catalog.go stays upstream's.
// Package-level vars are initialized before init() runs, and no package-level
// initializer reads the registry, so the row is present before any caller
// (config loading, schema generation, the DSL) can observe the registry.
func init() {
	decisionAlgorithmRegistry = append(decisionAlgorithmRegistry, decisionAlgorithmRegistryEntry{
		Catalog: AlgorithmCatalogEntry{
			Type:        RaylineARCAlgorithmType,
			DisplayName: "Rayline ARC",
			Description: "Select an arm with a trained Rayline ARC artifact or policy service.",
			Tier:        "experimental",
			Execution:   AlgorithmExecutionSelector,
			ConfigField: "rayline_arc",
		},
		IsConfigured: func(config *AlgorithmConfig) bool { return config.RaylineARC != nil },
	})
}
