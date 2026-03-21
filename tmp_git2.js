const { execSync } = require('child_process');
try {
  console.log(execSync('git add src/cache_util.ts src/engine.ts').toString());
  console.log(execSync('git commit -m "feat(security): verify mlc-chat-config and tokenizer integrity"').toString());
  console.log("Done");
} catch (e) {
  console.log("ERROR:", e.message);
}
