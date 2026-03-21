const { execSync } = require('child_process');
try {
  console.log("Branching fix-xss-robustly...");
  console.log(execSync('git checkout -b fix-xss-robustly').toString());
  console.log(execSync('git add src/utils.ts').toString());
  console.log(execSync('git commit -m "fix(sec): robust XSS filtering in sanitizeString"').toString());
  console.log("Checking out main...");
  console.log(execSync('git checkout main').toString());
  console.log("Branching feat-config-integrity...");
  console.log(execSync('git checkout -b feat-config-integrity').toString());
} catch (e) {
  console.log("ERROR:", e.message);
  console.log("STDOUT:", e.stdout ? e.stdout.toString() : '');
  console.log("STDERR:", e.stderr ? e.stderr.toString() : '');
}
