package logging_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The content-logging policy (issue #2464) says a diagnostic may carry
// correlation metadata, counts, timing, model identity and approved hashes,
// never prompt, response, query, tool or provider content. The policy helper
// (logging.ContentDescriptor) existed well before every caller adopted it, and
// the gap was invisible: each leak was one plausible-looking format argument,
// and log level is not a confidentiality boundary, so a Debugf dump of a prompt
// counts. These tests re-derive the audit from source on every run, so a
// reintroduced leak fails here rather than in a later retention review.
//
// Both tests are deliberately narrow. They name the variables that were
// actually found carrying content, rather than every identifier that might, so
// that the failure message is specific and the check does not become noise that
// gets skipped. A genuinely content-free use of one of these names is expected
// to be added to the exemption list with a reason.

// contentBearingArgs are expression suffixes that denote request or provider
// content: the user's turn, a model completion, routing reasoning over the
// prompt, or a provider body. A log or span argument ending in one of these is
// a leak unless it is wrapped in a descriptor.
var contentBearingArgs = []string{
	"ctx.RequestQuery",
	"span.Text",
	"s.Text",
	"result.Thinking",
	"response.Thinking",
	"string(body)",
	"string(requestBody)",
	"string(responseBody)",
}

// exemptions are expression/file pairs that match a pattern above but are not
// content. Keep the reason with the entry.
var exemptions = map[string]string{
	// Storage payloads, not logs: these write the cache entry itself, which is
	// the store's purpose. Retention there is controlled by disabling the
	// store, not by redacting the write.
	"pkg/cache/qdrant_cache.go":   "cache entry payload, not a log",
	"pkg/cache/redis_cache.go":    "cache entry hash fields, not a log",
	"pkg/cache/valkey_cache.go":   "cache entry hash fields, not a log",
	"pkg/cache/milvus_cache.go":   "cache entry columns, not a log",
	"pkg/cache/inmemory_cache.go": "cache entry struct, not a log",
	"pkg/cache/hybrid_cache.go":   "cache entry delegation, not a log",
}

// TestNoContentInLogCalls walks every non-test .go file under pkg/ and fails on
// a logging call whose arguments include raw content.
func TestNoContentInLogCalls(t *testing.T) {
	forEachCall(t, func(t *testing.T, path string, call *ast.CallExpr, fset *token.FileSet) {
		if !isLoggingCall(call) {
			return
		}
		// Check both the printf-style arguments and the values of a
		// structured event's map literal: logging.LogEvent carries its
		// content inside map[string]interface{}, which is where the
		// cache_hit and cache_entry_added leaks lived.
		for _, arg := range leafArgs(call.Args) {
			expr := exprString(fset, arg)
			if wrapsInDescriptor(expr) {
				continue
			}
			for _, bad := range contentBearingArgs {
				if expr != bad {
					continue
				}
				if _, ok := exemptions[path]; ok {
					continue
				}
				t.Errorf(
					"%s:%d: log argument %q passes content verbatim; wrap it in logging.ContentDescriptor",
					path, fset.Position(arg.Pos()).Line, expr,
				)
			}
		}
	})
}

// TestNoContentInSpanAttributes is the same check for OpenTelemetry span
// attributes. A span attribute leaves the process through the OTLP exporter
// (or stdout), so it is a retention surface in exactly the way a log line is,
// and the canonical configuration ships tracing enabled.
func TestNoContentInSpanAttributes(t *testing.T) {
	forEachCall(t, func(t *testing.T, path string, call *ast.CallExpr, fset *token.FileSet) {
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "String" {
			return
		}
		if pkg, ok := selector.X.(*ast.Ident); !ok || pkg.Name != "attribute" {
			return
		}
		for _, arg := range call.Args {
			expr := exprString(fset, arg)
			if wrapsInDescriptor(expr) {
				continue
			}
			for _, bad := range contentBearingArgs {
				if expr == bad {
					t.Errorf(
						"%s:%d: span attribute carries content %q; wrap it in logging.ContentDescriptor",
						path, fset.Position(arg.Pos()).Line, expr,
					)
				}
			}
		}
	})
}

// leafArgs flattens call arguments, descending into composite literals so the
// values of a structured event's field map are checked individually.
func leafArgs(args []ast.Expr) []ast.Expr {
	var leaves []ast.Expr
	for _, arg := range args {
		composite, ok := arg.(*ast.CompositeLit)
		if !ok {
			leaves = append(leaves, arg)
			continue
		}
		for _, element := range composite.Elts {
			if pair, ok := element.(*ast.KeyValueExpr); ok {
				leaves = append(leaves, pair.Value)
				continue
			}
			leaves = append(leaves, element)
		}
	}
	return leaves
}

// wrapsInDescriptor reports whether an argument is already redacted.
func wrapsInDescriptor(expr string) bool {
	return strings.Contains(expr, "ContentDescriptor")
}

// isLoggingCall reports whether a call writes to the log. It covers the
// logging package's printf-style and structured entry points, and the stdlib
// and fmt sinks that would bypass it.
func isLoggingCall(call *ast.CallExpr) bool {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := selector.X.(*ast.Ident)
	if !ok {
		return false
	}
	name := selector.Sel.Name
	switch pkg.Name {
	case "logging":
		return strings.HasSuffix(name, "f") || strings.HasSuffix(name, "Event")
	case "log":
		return strings.HasPrefix(name, "Print") || strings.HasPrefix(name, "Fatal")
	case "fmt":
		return strings.HasPrefix(name, "Print")
	}
	return false
}

// forEachCall parses every non-test .go file under pkg/ and visits each call.
func forEachCall(
	t *testing.T,
	visit func(t *testing.T, path string, call *ast.CallExpr, fset *token.FileSet),
) {
	t.Helper()
	root := pkgRoot(t)
	fset := token.NewFileSet()
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		file, parseErr := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if parseErr != nil {
			// A file this walker cannot parse is not a policy failure; the
			// compiler is the authority on syntax.
			return nil
		}
		rel, relErr := filepath.Rel(filepath.Dir(root), path)
		if relErr != nil {
			rel = path
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if call, ok := node.(*ast.CallExpr); ok {
				visit(t, rel, call, fset)
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatalf("walk pkg/: %v", err)
	}
}

// pkgRoot locates the pkg/ directory this test lives under.
func pkgRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	// .../pkg/observability/logging -> .../pkg
	root := filepath.Join(dir, "..", "..")
	if base := filepath.Base(root); base != "pkg" {
		t.Fatalf("expected to resolve pkg/, got %q", root)
	}
	return root
}

// exprString renders an expression as source text for comparison.
func exprString(fset *token.FileSet, expr ast.Expr) string {
	start := fset.Position(expr.Pos())
	end := fset.Position(expr.End())
	data, err := os.ReadFile(start.Filename)
	if err != nil || end.Offset > len(data) {
		return ""
	}
	return strings.TrimSpace(string(data[start.Offset:end.Offset]))
}
