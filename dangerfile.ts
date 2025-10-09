import { danger, warn, message, fail } from "danger";

// Get PR info
const pr = danger.github.pr;
const files = danger.git.modified_files.concat(danger.git.created_files);

// Big PR warning
const size = pr.additions + pr.deletions;
if (size > 600) {
  warn(`🚨 Large PR detected (${size} LOC). Consider splitting for easier review and reduced risk.`);
}

// Python debug statements and TODO markers
const pythonFiles = files.filter((f: string) => f.endsWith(".py"));
if (pythonFiles.length > 0) {
  warn(`🐛 Python files modified. Please manually check for debug statements or TODO markers in: ${pythonFiles.join(", ")}`);
}

// Synndicate-specific: Critical system files require extra scrutiny
const criticalFiles = files.filter((f: string) => 
  f.includes("core/orchestrator.py") || 
  f.includes("api/server.py") ||
  f.includes("config/settings.py") ||
  f.includes("observability/distributed_tracing.py") ||
  f.includes("core/determinism.py")
);

if (criticalFiles.length) {
  message(`⚠️ Critical system files modified. Extra review recommended:\n${criticalFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// Observability changes require metrics validation
const obsFiles = files.filter((f: string) => 
  f.includes("observability/") || 
  f.includes("metrics.py") ||
  f.includes("logging.py") ||
  f.includes("tracing.py")
);

if (obsFiles.length) {
  message(`🔍 Observability changes detected. Verify metrics, logging, and tracing still work correctly:\n${obsFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// API/Security changes require security review
const apiSecurityFiles = files.filter((f: string) => 
  f.includes("api/") || 
  f.includes("auth.py") ||
  f.includes("security") ||
  f.includes("jwt") ||
  f.startsWith("src/synndicate/api/")
);

if (apiSecurityFiles.length) {
  message(`🔒 API/Security changes detected. Security review recommended:\n${apiSecurityFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// RAG/Model changes require validation
const ragModelFiles = files.filter((f: string) => 
  f.includes("rag/") || 
  f.includes("models/") ||
  f.includes("agents/") ||
  f.includes("retrieval") ||
  f.includes("vectorstore")
);

if (ragModelFiles.length) {
  message(`🤖 RAG/Model changes detected. Validate model behavior and retrieval accuracy:\n${ragModelFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// Configuration changes are sensitive
const configFiles = files.filter((f: string) => 
  f.includes("config/") ||
  f.includes("settings.py") ||
  f.includes("pyproject.toml") ||
  f.includes("requirements") ||
  f.includes("docker") ||
  f.includes("compose")
);

if (configFiles.length) {
  message(`⚙️ Configuration changes detected. Verify environment compatibility:\n${configFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// Test coverage expectations
const touchedSrc = files.some((f: string) => f.startsWith("src/synndicate/"));
const touchedTests = files.some((f: string) => f.startsWith("tests/"));

if (touchedSrc && !touchedTests) {
  message("🧪 Source code changes detected without corresponding test updates. Consider adding/updating tests.");
}

// Type safety reminders for critical areas
const typeFiles = files.filter((f: string) => 
  f.endsWith(".py") && (
    f.includes("core/") ||
    f.includes("api/") ||
    f.includes("observability/")
  )
);

if (typeFiles.length) {
  message("🔬 Changes to type-critical areas detected. Ensure MyPy strict mode passes and type annotations are complete.");
}

// Documentation updates for public interfaces
const publicInterfaceFiles = files.filter((f: string) => 
  f.includes("api/") && f.endsWith(".py")
);

if (publicInterfaceFiles.length) {
  const hasDocUpdates = files.some((f: string) => 
    f.includes("docs/") || 
    f.includes("README") ||
    f.includes("CHANGELOG")
  );
  
  if (!hasDocUpdates) {
    message("📚 API changes detected. Consider updating documentation if public interfaces changed.");
  }
}

// Security-sensitive file patterns
const securitySensitiveFiles = files.filter((f: string) => 
  f.endsWith(".py") &&
  (f.includes("auth") || f.includes("security") || f.includes("jwt") || f.includes("crypto")) &&
  !f.includes("test")
);

if (securitySensitiveFiles.length) {
  message(`🔐 Security-sensitive files modified. Extra security review required:\n${securitySensitiveFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// Performance-critical areas
const perfCriticalFiles = files.filter((f: string) => 
  f.includes("orchestrator.py") ||
  f.includes("retrieval") ||
  f.includes("vectorstore") ||
  f.includes("agents/")
);

if (perfCriticalFiles.length) {
  message(`⚡ Performance-critical areas modified. Consider benchmarking:\n${perfCriticalFiles.map((f: string) => `- ${f}`).join('\n')}`);
}

// PR title and description quality
if (!pr.title || pr.title.length < 10) {
  fail("❌ PR title is too short. Please provide a descriptive title.");
}

if (!pr.body || pr.body.length < 20) {
  warn("📝 PR description is quite short. Consider adding more context about the changes.");
}

// Changelog reminder for significant changes
if (size > 100 && !files.some((f: string) => f.includes("CHANGELOG"))) {
  message("📋 Significant changes detected. Consider updating CHANGELOG.md if this affects users or deployment.");
}
